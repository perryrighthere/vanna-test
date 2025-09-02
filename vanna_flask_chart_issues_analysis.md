# Vanna Flask 图表生成问题分析报告

## 概述

本报告深入分析了 Vanna 框架在 Flask 环境下图表生成经常出现问题的潜在原因。通过对代码库的详细检查，识别了多个可能导致图表生成失败的关键问题点。

## 问题分析

### 1. LLM 生成的 Plotly 代码质量问题

**问题位置**: `src/vanna/base/base.py:735-757`

**详细描述**:
- LLM 基于简单的提示词生成 Plotly 代码，可能包含语法错误、导入错误或逻辑错误
- 提示词相对简单，无法处理复杂的数据结构和边缘情况
- 代码提取使用正则表达式 (`src/vanna/base/base.py:713`)，可能匹配错误的代码块

**相关代码**:
```python
def generate_plotly_code(self, question: str = None, sql: str = None, df_metadata: str = None, **kwargs) -> str:
    # 提示词构建
    message_log = [
        self.system_message(system_msg),
        self.user_message(
            "Can you generate the Python plotly code to chart the results of the dataframe? "
            "Assume the data is in a pandas dataframe called 'df'. If there is only one value "
            "in the dataframe, use an Indicator. Respond with only Python code. "
            "Do not answer with any explanations -- just the code."
        ),
    ]
    plotly_code = self.submit_prompt(message_log, kwargs=kwargs)
    return self._sanitize_plotly_code(self._extract_python_code(plotly_code))
```

**可能现象**:
- 生成的代码包含未定义的变量
- 图表类型选择不当
- 代码语法错误导致执行失败

### 2. 危险的代码执行机制

**问题位置**: `src/vanna/base/base.py:2088`

**详细描述**:
- 使用 `exec()` 直接执行 LLM 生成的未验证代码存在安全风险
- Flask 环境可能对某些操作有限制，导致代码执行失败
- 执行环境可能缺少必要的依赖或上下文

**相关代码**:
```python
def get_plotly_figure(self, plotly_code: str, df: pd.DataFrame, dark_mode: bool = True) -> plotly.graph_objs.Figure:
    ldict = {"df": df, "px": px, "go": go}
    try:
        exec(plotly_code, globals(), ldict)  # 危险的代码执行
        fig = ldict.get("fig", None)
    except Exception as e:
        # 回退到默认图表类型
        ...
```

**潜在风险**:
- 任意代码执行安全漏洞
- 运行时错误导致图表生成失败
- 内存泄漏或性能问题

### 3. 数据类型转换问题

**问题位置**: `src/vanna/flask/__init__.py:688`

**详细描述**:
- DataFrame 的数据类型信息传递给 LLM 时可能不够详细
- 某些数据类型（如时间戳、复杂对象）可能导致 Plotly 代码生成错误
- 数据中存在 NaN 或 None 值可能导致绘图失败

**相关代码**:
```python
@self.flask_app.route("/api/v0/generate_plotly_figure", methods=["GET"])
@self.requires_auth
@self.requires_cache(["df", "question", "sql"])
def generate_plotly_figure(user: any, id: str, df, question, sql):
    code = vn.generate_plotly_code(
        question=question,
        sql=sql,
        df_metadata=f"Running df.dtypes gives:\n {df.dtypes}",  # 数据类型信息过于简单
    )
```

**可能问题**:
- 数据类型信息不足，LLM 无法生成合适的图表代码
- 特殊数据类型（如日期时间）处理不当
- 缺少数据范围和分布信息

### 4. JSON 序列化问题

**问题位置**: `src/vanna/flask/__init__.py:693`

**详细描述**:
- Plotly 图表转换为 JSON 时可能失败
- 某些 Plotly 图表类型可能不支持完整的 JSON 序列化
- 大型数据集可能导致 JSON 过大，影响传输

**相关代码**:
```python
fig = vn.get_plotly_figure(plotly_code=code, df=df, dark_mode=False)
fig_json = fig.to_json()  # 可能失败的序列化
self.cache.set(id=id, field="fig_json", value=fig_json)
```

**潜在问题**:
- JSON 序列化异常
- 图表数据过大导致性能问题
- 特殊字符或数据类型序列化错误

### 5. 错误处理机制不完善

**问题位置**: `src/vanna/base/base.py:2091-2110`

**详细描述**:
- 当 Plotly 代码执行失败时，回退机制选择的默认图表类型可能不合适
- 错误信息不够详细，难以调试和定位问题
- 回退逻辑过于简单，无法处理复杂场景

**相关代码**:
```python
except Exception as e:
    # 简单的回退机制
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    if len(numeric_cols) >= 2:
        fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1])
    elif len(numeric_cols) == 1 and len(categorical_cols) >= 1:
        fig = px.bar(df, x=categorical_cols[0], y=numeric_cols[0])
    # ...
```

**局限性**:
- 回退逻辑可能选择不合适的图表类型
- 错误信息丢失，难以定位根本原因
- 无法处理特殊数据结构

### 6. Flask 环境特殊限制

**问题位置**: `src/vanna/flask/__init__.py:692`

**详细描述**:
- Flask 应用可能运行在受限环境中，限制某些操作
- `dark_mode=False` 硬编码，可能影响某些图表的样式渲染
- Web 环境的内存和执行时间限制可能导致大型图表生成失败

**相关代码**:
```python
fig = vn.get_plotly_figure(plotly_code=code, df=df, dark_mode=False)  # 硬编码主题
```

**限制因素**:
- Web 服务器资源限制
- JavaScript 执行环境差异
- 浏览器兼容性问题

## 建议解决方案

### 1. 改进代码生成提示词

**优化策略**:
- 添加更详细的数据类型和结构信息
- 包含常见错误处理指令
- 提供具体的 Plotly 代码示例和最佳实践

**实现建议**:
```python
def generate_plotly_code(self, question: str = None, sql: str = None, df_metadata: str = None, **kwargs) -> str:
    # 增强的数据描述
    enhanced_metadata = f"""
    DataFrame Info:
    - Shape: {df.shape}
    - Columns: {list(df.columns)}
    - Data types: {df.dtypes.to_dict()}
    - Null values: {df.isnull().sum().to_dict()}
    - Sample data: {df.head(3).to_dict()}
    """
    
    # 更详细的提示词
    enhanced_prompt = """
    Generate Python plotly code with error handling:
    1. Check for null values and handle appropriately
    2. Ensure proper data type conversion
    3. Add try-catch blocks for robustness
    4. Use appropriate chart types based on data characteristics
    """
```

### 2. 增强代码验证机制

**验证步骤**:
- 语法检查：使用 `ast.parse()` 验证生成代码的语法正确性
- 变量检查：确保代码中引用的变量存在
- 函数检查：验证使用的 Plotly 函数是否存在且参数正确

**实现示例**:
```python
import ast

def validate_plotly_code(self, code: str) -> bool:
    try:
        # 语法检查
        ast.parse(code)
        
        # 变量和函数检查
        allowed_names = {'df', 'px', 'go', 'fig'}
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id not in allowed_names:
                return False
                
        return True
    except:
        return False
```

### 3. 完善错误处理和日志

**改进措施**:
- 详细的错误日志记录
- 分级错误处理（语法错误、运行时错误、数据错误）
- 用户友好的错误信息展示

**实现方案**:
```python
import logging

def get_plotly_figure_enhanced(self, plotly_code: str, df: pd.DataFrame, dark_mode: bool = True):
    logger = logging.getLogger(__name__)
    
    try:
        # 代码验证
        if not self.validate_plotly_code(plotly_code):
            raise ValueError("Generated code failed validation")
            
        # 安全执行
        ldict = {"df": df, "px": px, "go": go}
        exec(plotly_code, {"__builtins__": {}}, ldict)
        
        fig = ldict.get("fig", None)
        if fig is None:
            raise ValueError("No figure object created")
            
        return fig
        
    except Exception as e:
        logger.error(f"Plotly code execution failed: {str(e)}")
        logger.error(f"Failed code: {plotly_code}")
        
        # 智能回退机制
        return self.create_fallback_chart(df, str(e))
```

### 4. 数据预处理优化

**预处理步骤**:
- 数据类型标准化
- 异常值和空值处理
- 数据规模控制

**实现建议**:
```python
def preprocess_data_for_chart(self, df: pd.DataFrame) -> pd.DataFrame:
    # 限制数据规模
    if len(df) > 10000:
        df = df.sample(n=10000)
    
    # 处理空值
    df = df.dropna(subset=df.select_dtypes(include=['number']).columns.tolist()[:2])
    
    # 数据类型转换
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                pass
                
    return df
```

### 5. 安全性改进

**安全措施**:
- 限制可执行的操作和导入
- 使用受限的执行环境
- 代码沙箱机制

**实现方案**:
```python
def safe_exec(self, code: str, local_vars: dict):
    # 受限的内置函数
    safe_builtins = {
        '__builtins__': {
            'len': len, 'range': range, 'enumerate': enumerate,
            'zip': zip, 'map': map, 'filter': filter,
        }
    }
    
    # 超时保护
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Code execution timeout")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)  # 30秒超时
    
    try:
        exec(code, safe_builtins, local_vars)
    finally:
        signal.alarm(0)
```

## 总结

Vanna 在 Flask 环境下的图表生成问题主要源于：

1. **代码生成质量**：LLM 生成的代码质量不稳定
2. **执行安全性**：直接执行未验证的代码存在风险
3. **错误处理**：回退机制和错误信息不够完善
4. **数据处理**：数据预处理和类型转换不充分
5. **环境限制**：Flask 和 Web 环境的特殊限制

通过实施上述建议的解决方案，可以显著提高图表生成的稳定性和成功率，同时增强系统的安全性和用户体验。

---

*分析报告生成时间：2025-01-21*  
*基于 Vanna 版本：0.7.9*