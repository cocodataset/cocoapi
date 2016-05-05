package = "coco"
version = "scm-1"

source = {
  url = "git://github.com/pdollar/coco.git"
}

description = {
  summary = "Interface for accessing the Microsoft COCO dataset",
  detailed = "See http://mscoco.org/ for more details",
  homepage = "https://github.com/pdollar/coco",
  license = "Simplified BSD"
}

dependencies = {
  "lua >= 5.1",
  "torch >= 7.0",
  "lua-cjson"
}

build = {
  type = "builtin",
  modules = {
    ["coco.env"] = "LuaApi/env.lua",
    ["coco.init"] = "LuaApi/init.lua",
    ["coco.maskapi"] = "LuaApi/MaskApi.lua",
    ["coco.cocoapi"] = "LuaApi/CocoApi.lua",
    libmaskapi = {
      sources = { "common/maskApi.c" },
      incdirs = { "common/" }
    }
  }
}

-- luarocks make LuaAPI/rocks/coco-scm-1.rockspec
-- https://github.com/pdollar/coco/raw/master/LuaAPI/rocks/coco-scm-1.rockspec
