-- xmake.lua
-- 项目配置文件

-- 设置工程名称
set_project("stschedule")

-- 设置 xmake 最小版本
set_xmakever("2.5.1")

-- 设置 C++ 标准
set_languages("c++17")

-- 添加编译选项
add_cxxflags("-Wall", "-Wextra")

-- 定义构建模式
add_rules("mode.debug", "mode.release")
add_rules("plugin.compile_commands.autoupdate", {outputdir = ".vscode"})

-- 定义目标
target("stschedule")
    -- 设置目标类型为可执行文件
    set_kind("binary")
    
    -- 添加头文件搜索路径
    add_includedirs("include")
    
    -- 添加源文件
    add_files("src/*.cpp")
    add_files("src/nns/*.cpp")
    add_files("src/json/*.cpp")
    add_files("src/spatial_mapping/*.cpp")
    
    -- 添加链接库
    add_links("pthread", "m", "stdc++")
    
    -- 设置输出目录
    set_targetdir("$(builddir)")
    
    -- debug 模式配置
    if is_mode("debug") then
        add_defines("DEBUG")
        set_symbols("debug")
        set_optimize("none")
    end
    
    -- release 模式配置
    if is_mode("release") then
        set_optimize("aggressive")
        set_strip("all")
    end
    
    -- perf 模式配置（用于性能分析）
    -- 可以通过 xmake f --perf=y 来启用性能分析符号
