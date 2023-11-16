var t =
    "undefined" != typeof globalThis
      ? globalThis
      : "undefined" != typeof window
      ? window
      : "undefined" != typeof global
      ? global
      : "undefined" != typeof self
      ? self
      : {},
  e = function (t) {
    return t && t.Math == Math && t;
  },
  n =
    e("object" == typeof globalThis && globalThis) ||
    e("object" == typeof window && window) ||
    e("object" == typeof self && self) ||
    e("object" == typeof t && t) ||
    (function () {
      return this;
    })() ||
    Function("return this")(),
  r = {},
  o = function (t) {
    try {
      return !!t();
    } catch (t) {
      return !0;
    }
  },
  i = !o(function () {
    return (
      7 !=
      Object.defineProperty({}, 1, {
        get: function () {
          return 7;
        },
      })[1]
    );
  }),
  a = {},
  c = {}.propertyIsEnumerable,
  u = Object.getOwnPropertyDescriptor,
  f = u && !c.call({ 1: 2 }, 1);
a.f = f
  ? function (t) {
      var e = u(this, t);
      return !!e && e.enumerable;
    }
  : c;
var s = function (t, e) {
    return {
      enumerable: !(1 & t),
      configurable: !(2 & t),
      writable: !(4 & t),
      value: e,
    };
  },
  l = {}.toString,
  p = function (t) {
    return l.call(t).slice(8, -1);
  },
  h = p,
  v = "".split,
  d = o(function () {
    return !Object("z").propertyIsEnumerable(0);
  })
    ? function (t) {
        return "String" == h(t) ? v.call(t, "") : Object(t);
      }
    : Object,
  y = function (t) {
    if (null == t) throw TypeError("Can't call method on " + t);
    return t;
  },
  g = d,
  m = y,
  b = function (t) {
    return g(m(t));
  },
  w = function (t) {
    return "object" == typeof t ? null !== t : "function" == typeof t;
  },
  j = w,
  x = function (t, e) {
    if (!j(t)) return t;
    var n, r;
    if (e && "function" == typeof (n = t.toString) && !j((r = n.call(t))))
      return r;
    if ("function" == typeof (n = t.valueOf) && !j((r = n.call(t)))) return r;
    if (!e && "function" == typeof (n = t.toString) && !j((r = n.call(t))))
      return r;
    throw TypeError("Can't convert object to primitive value");
  },
  O = y,
  E = function (t) {
    return Object(O(t));
  },
  S = E,
  T = {}.hasOwnProperty,
  P = function (t, e) {
    return T.call(S(t), e);
  },
  _ = w,
  k = n.document,
  L = _(k) && _(k.createElement),
  M = function (t) {
    return L ? k.createElement(t) : {};
  },
  A = M,
  I =
    !i &&
    !o(function () {
      return (
        7 !=
        Object.defineProperty(A("div"), "a", {
          get: function () {
            return 7;
          },
        }).a
      );
    }),
  R = i,
  C = a,
  N = s,
  F = b,
  J = x,
  D = P,
  $ = I,
  G = Object.getOwnPropertyDescriptor;
r.f = R
  ? G
  : function (t, e) {
      if (((t = F(t)), (e = J(e, !0)), $))
        try {
          return G(t, e);
        } catch (t) {}
      if (D(t, e)) return N(!C.f.call(t, e), t[e]);
    };
var H = {},
  z = w,
  W = function (t) {
    if (!z(t)) throw TypeError(String(t) + " is not an object");
    return t;
  },
  q = i,
  U = I,
  K = W,
  Q = x,
  X = Object.defineProperty;
H.f = q
  ? X
  : function (t, e, n) {
      if ((K(t), (e = Q(e, !0)), K(n), U))
        try {
          return X(t, e, n);
        } catch (t) {}
      if ("get" in n || "set" in n) throw TypeError("Accessors not supported");
      return "value" in n && (t[e] = n.value), t;
    };
var Y = H,
  B = s,
  V = i
    ? function (t, e, n) {
        return Y.f(t, e, B(1, n));
      }
    : function (t, e, n) {
        return (t[e] = n), t;
      },
  Z = { exports: {} },
  tt = n,
  et = V,
  nt = function (t, e) {
    try {
      et(tt, t, e);
    } catch (n) {
      tt[t] = e;
    }
    return e;
  },
  rt = nt,
  ot = n["__core-js_shared__"] || rt("__core-js_shared__", {}),
  it = ot,
  at = Function.toString;
"function" != typeof it.inspectSource &&
  (it.inspectSource = function (t) {
    return at.call(t);
  });
var ct = it.inspectSource,
  ut = ct,
  ft = n.WeakMap,
  st = "function" == typeof ft && /native code/.test(ut(ft)),
  lt = { exports: {} },
  pt = ot;
(lt.exports = function (t, e) {
  return pt[t] || (pt[t] = void 0 !== e ? e : {});
})("versions", []).push({
  version: "3.12.1",
  mode: "global",
  copyright: "© 2021 Denis Pushkarev (zloirock.ru)",
});
var ht,
  vt,
  dt,
  yt = 0,
  gt = Math.random(),
  mt = function (t) {
    return (
      "Symbol(" +
      String(void 0 === t ? "" : t) +
      ")_" +
      (++yt + gt).toString(36)
    );
  },
  bt = lt.exports,
  wt = mt,
  jt = bt("keys"),
  xt = function (t) {
    return jt[t] || (jt[t] = wt(t));
  },
  Ot = {},
  Et = st,
  St = w,
  Tt = V,
  Pt = P,
  _t = ot,
  kt = xt,
  Lt = Ot,
  Mt = n.WeakMap;
if (Et || _t.state) {
  var At = _t.state || (_t.state = new Mt()),
    It = At.get,
    Rt = At.has,
    Ct = At.set;
  (ht = function (t, e) {
    if (Rt.call(At, t)) throw new TypeError("Object already initialized");
    return (e.facade = t), Ct.call(At, t, e), e;
  }),
    (vt = function (t) {
      return It.call(At, t) || {};
    }),
    (dt = function (t) {
      return Rt.call(At, t);
    });
} else {
  var Nt = kt("state");
  (Lt[Nt] = !0),
    (ht = function (t, e) {
      if (Pt(t, Nt)) throw new TypeError("Object already initialized");
      return (e.facade = t), Tt(t, Nt, e), e;
    }),
    (vt = function (t) {
      return Pt(t, Nt) ? t[Nt] : {};
    }),
    (dt = function (t) {
      return Pt(t, Nt);
    });
}
var Ft = {
    set: ht,
    get: vt,
    has: dt,
    enforce: function (t) {
      return dt(t) ? vt(t) : ht(t, {});
    },
    getterFor: function (t) {
      return function (e) {
        var n;
        if (!St(e) || (n = vt(e)).type !== t)
          throw TypeError("Incompatible receiver, " + t + " required");
        return n;
      };
    },
  },
  Jt = n,
  Dt = V,
  $t = P,
  Gt = nt,
  Ht = ct,
  zt = Ft.get,
  Wt = Ft.enforce,
  qt = String(String).split("String");
(Z.exports = function (t, e, n, r) {
  var o,
    i = !!r && !!r.unsafe,
    a = !!r && !!r.enumerable,
    c = !!r && !!r.noTargetGet;
  "function" == typeof n &&
    ("string" != typeof e || $t(n, "name") || Dt(n, "name", e),
    (o = Wt(n)).source || (o.source = qt.join("string" == typeof e ? e : ""))),
    t !== Jt
      ? (i ? !c && t[e] && (a = !0) : delete t[e], a ? (t[e] = n) : Dt(t, e, n))
      : a
      ? (t[e] = n)
      : Gt(e, n);
})(Function.prototype, "toString", function () {
  return ("function" == typeof this && zt(this).source) || Ht(this);
});
var Ut = n,
  Kt = n,
  Qt = function (t) {
    return "function" == typeof t ? t : void 0;
  },
  Xt = function (t, e) {
    return arguments.length < 2
      ? Qt(Ut[t]) || Qt(Kt[t])
      : (Ut[t] && Ut[t][e]) || (Kt[t] && Kt[t][e]);
  },
  Yt = {},
  Bt = Math.ceil,
  Vt = Math.floor,
  Zt = function (t) {
    return isNaN((t = +t)) ? 0 : (t > 0 ? Vt : Bt)(t);
  },
  te = Zt,
  ee = Math.min,
  ne = function (t) {
    return t > 0 ? ee(te(t), 9007199254740991) : 0;
  },
  re = Zt,
  oe = Math.max,
  ie = Math.min,
  ae = b,
  ce = ne,
  ue = function (t, e) {
    var n = re(t);
    return n < 0 ? oe(n + e, 0) : ie(n, e);
  },
  fe = function (t) {
    return function (e, n, r) {
      var o,
        i = ae(e),
        a = ce(i.length),
        c = ue(r, a);
      if (t && n != n) {
        for (; a > c; ) if ((o = i[c++]) != o) return !0;
      } else
        for (; a > c; c++) if ((t || c in i) && i[c] === n) return t || c || 0;
      return !t && -1;
    };
  },
  se = { includes: fe(!0), indexOf: fe(!1) },
  le = P,
  pe = b,
  he = se.indexOf,
  ve = Ot,
  de = function (t, e) {
    var n,
      r = pe(t),
      o = 0,
      i = [];
    for (n in r) !le(ve, n) && le(r, n) && i.push(n);
    for (; e.length > o; ) le(r, (n = e[o++])) && (~he(i, n) || i.push(n));
    return i;
  },
  ye = [
    "constructor",
    "hasOwnProperty",
    "isPrototypeOf",
    "propertyIsEnumerable",
    "toLocaleString",
    "toString",
    "valueOf",
  ],
  ge = de,
  me = ye.concat("length", "prototype");
Yt.f =
  Object.getOwnPropertyNames ||
  function (t) {
    return ge(t, me);
  };
var be = {};
be.f = Object.getOwnPropertySymbols;
var we = Yt,
  je = be,
  xe = W,
  Oe =
    Xt("Reflect", "ownKeys") ||
    function (t) {
      var e = we.f(xe(t)),
        n = je.f;
      return n ? e.concat(n(t)) : e;
    },
  Ee = P,
  Se = Oe,
  Te = r,
  Pe = H,
  _e = o,
  ke = /#|\.prototype\./,
  Le = function (t, e) {
    var n = Ae[Me(t)];
    return n == Re || (n != Ie && ("function" == typeof e ? _e(e) : !!e));
  },
  Me = (Le.normalize = function (t) {
    return String(t).replace(ke, ".").toLowerCase();
  }),
  Ae = (Le.data = {}),
  Ie = (Le.NATIVE = "N"),
  Re = (Le.POLYFILL = "P"),
  Ce = Le,
  Ne = n,
  Fe = r.f,
  Je = V,
  De = Z.exports,
  $e = nt,
  Ge = function (t, e) {
    for (var n = Se(e), r = Pe.f, o = Te.f, i = 0; i < n.length; i++) {
      var a = n[i];
      Ee(t, a) || r(t, a, o(e, a));
    }
  },
  He = Ce,
  ze = function (t, e) {
    var n,
      r,
      o,
      i,
      a,
      c = t.target,
      u = t.global,
      f = t.stat;
    if ((n = u ? Ne : f ? Ne[c] || $e(c, {}) : (Ne[c] || {}).prototype))
      for (r in e) {
        if (
          ((i = e[r]),
          (o = t.noTargetGet ? (a = Fe(n, r)) && a.value : n[r]),
          !He(u ? r : c + (f ? "." : "#") + r, t.forced) && void 0 !== o)
        ) {
          if (typeof i == typeof o) continue;
          Ge(i, o);
        }
        (t.sham || (o && o.sham)) && Je(i, "sham", !0), De(n, r, i, t);
      }
  },
  We = de,
  qe = ye,
  Ue =
    Object.keys ||
    function (t) {
      return We(t, qe);
    },
  Ke = i,
  Qe = o,
  Xe = Ue,
  Ye = be,
  Be = a,
  Ve = E,
  Ze = d,
  tn = Object.assign,
  en = Object.defineProperty,
  nn =
    !tn ||
    Qe(function () {
      if (
        Ke &&
        1 !==
          tn(
            { b: 1 },
            tn(
              en({}, "a", {
                enumerable: !0,
                get: function () {
                  en(this, "b", { value: 3, enumerable: !1 });
                },
              }),
              { b: 2 },
            ),
          ).b
      )
        return !0;
      var t = {},
        e = {},
        n = Symbol(),
        r = "abcdefghijklmnopqrst";
      return (
        (t[n] = 7),
        r.split("").forEach(function (t) {
          e[t] = t;
        }),
        7 != tn({}, t)[n] || Xe(tn({}, e)).join("") != r
      );
    })
      ? function (t, e) {
          for (
            var n = Ve(t), r = arguments.length, o = 1, i = Ye.f, a = Be.f;
            r > o;

          )
            for (
              var c,
                u = Ze(arguments[o++]),
                f = i ? Xe(u).concat(i(u)) : Xe(u),
                s = f.length,
                l = 0;
              s > l;

            )
              (c = f[l++]), (Ke && !a.call(u, c)) || (n[c] = u[c]);
          return n;
        }
      : tn;
function rn(t, e) {
  var n = Object.keys(t);
  if (Object.getOwnPropertySymbols) {
    var r = Object.getOwnPropertySymbols(t);
    e &&
      (r = r.filter(function (e) {
        return Object.getOwnPropertyDescriptor(t, e).enumerable;
      })),
      n.push.apply(n, r);
  }
  return n;
}
function on(t) {
  for (var e = 1; e < arguments.length; e++) {
    var n = null != arguments[e] ? arguments[e] : {};
    e % 2
      ? rn(Object(n), !0).forEach(function (e) {
          un(t, e, n[e]);
        })
      : Object.getOwnPropertyDescriptors
      ? Object.defineProperties(t, Object.getOwnPropertyDescriptors(n))
      : rn(Object(n)).forEach(function (e) {
          Object.defineProperty(t, e, Object.getOwnPropertyDescriptor(n, e));
        });
  }
  return t;
}
function an(t, e, n, r, o, i, a) {
  try {
    var c = t[i](a),
      u = c.value;
  } catch (t) {
    return void n(t);
  }
  c.done ? e(u) : Promise.resolve(u).then(r, o);
}
function cn(t) {
  return function () {
    var e = this,
      n = arguments;
    return new Promise(function (r, o) {
      var i = t.apply(e, n);
      function a(t) {
        an(i, r, o, a, c, "next", t);
      }
      function c(t) {
        an(i, r, o, a, c, "throw", t);
      }
      a(void 0);
    });
  };
}
function un(t, e, n) {
  return (
    e in t
      ? Object.defineProperty(t, e, {
          value: n,
          enumerable: !0,
          configurable: !0,
          writable: !0,
        })
      : (t[e] = n),
    t
  );
}
function fn(t, e) {
  if (null == t) return {};
  var n,
    r,
    o = (function (t, e) {
      if (null == t) return {};
      var n,
        r,
        o = {},
        i = Object.keys(t);
      for (r = 0; r < i.length; r++)
        (n = i[r]), e.indexOf(n) >= 0 || (o[n] = t[n]);
      return o;
    })(t, e);
  if (Object.getOwnPropertySymbols) {
    var i = Object.getOwnPropertySymbols(t);
    for (r = 0; r < i.length; r++)
      (n = i[r]),
        e.indexOf(n) >= 0 ||
          (Object.prototype.propertyIsEnumerable.call(t, n) && (o[n] = t[n]));
  }
  return o;
}
function sn(t, e) {
  (null == e || e > t.length) && (e = t.length);
  for (var n = 0, r = new Array(e); n < e; n++) r[n] = t[n];
  return r;
}
function ln(t, e) {
  var n =
    ("undefined" != typeof Symbol && t[Symbol.iterator]) || t["@@iterator"];
  if (!n) {
    if (
      Array.isArray(t) ||
      (n = (function (t, e) {
        if (t) {
          if ("string" == typeof t) return sn(t, e);
          var n = Object.prototype.toString.call(t).slice(8, -1);
          return (
            "Object" === n && t.constructor && (n = t.constructor.name),
            "Map" === n || "Set" === n
              ? Array.from(t)
              : "Arguments" === n ||
                /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)
              ? sn(t, e)
              : void 0
          );
        }
      })(t)) ||
      (e && t && "number" == typeof t.length)
    ) {
      n && (t = n);
      var r = 0,
        o = function () {};
      return {
        s: o,
        n: function () {
          return r >= t.length ? { done: !0 } : { done: !1, value: t[r++] };
        },
        e: function (t) {
          throw t;
        },
        f: o,
      };
    }
    throw new TypeError(
      "Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.",
    );
  }
  var i,
    a = !0,
    c = !1;
  return {
    s: function () {
      n = n.call(t);
    },
    n: function () {
      var t = n.next();
      return (a = t.done), t;
    },
    e: function (t) {
      (c = !0), (i = t);
    },
    f: function () {
      try {
        a || null == n.return || n.return();
      } finally {
        if (c) throw i;
      }
    },
  };
}
ze(
  { target: "Object", stat: !0, forced: Object.assign !== nn },
  { assign: nn },
);
!(function (t) {
  var e = (function (t) {
    var e,
      n = Object.prototype,
      r = n.hasOwnProperty,
      o = "function" == typeof Symbol ? Symbol : {},
      i = o.iterator || "@@iterator",
      a = o.asyncIterator || "@@asyncIterator",
      c = o.toStringTag || "@@toStringTag";
    function u(t, e, n) {
      return (
        Object.defineProperty(t, e, {
          value: n,
          enumerable: !0,
          configurable: !0,
          writable: !0,
        }),
        t[e]
      );
    }
    try {
      u({}, "");
    } catch (t) {
      u = function (t, e, n) {
        return (t[e] = n);
      };
    }
    function f(t, e, n, r) {
      var o = e && e.prototype instanceof y ? e : y,
        i = Object.create(o.prototype),
        a = new _(r || []);
      return (
        (i._invoke = (function (t, e, n) {
          var r = l;
          return function (o, i) {
            if (r === h) throw new Error("Generator is already running");
            if (r === v) {
              if ("throw" === o) throw i;
              return L();
            }
            for (n.method = o, n.arg = i; ; ) {
              var a = n.delegate;
              if (a) {
                var c = S(a, n);
                if (c) {
                  if (c === d) continue;
                  return c;
                }
              }
              if ("next" === n.method) n.sent = n._sent = n.arg;
              else if ("throw" === n.method) {
                if (r === l) throw ((r = v), n.arg);
                n.dispatchException(n.arg);
              } else "return" === n.method && n.abrupt("return", n.arg);
              r = h;
              var u = s(t, e, n);
              if ("normal" === u.type) {
                if (((r = n.done ? v : p), u.arg === d)) continue;
                return { value: u.arg, done: n.done };
              }
              "throw" === u.type &&
                ((r = v), (n.method = "throw"), (n.arg = u.arg));
            }
          };
        })(t, n, a)),
        i
      );
    }
    function s(t, e, n) {
      try {
        return { type: "normal", arg: t.call(e, n) };
      } catch (t) {
        return { type: "throw", arg: t };
      }
    }
    t.wrap = f;
    var l = "suspendedStart",
      p = "suspendedYield",
      h = "executing",
      v = "completed",
      d = {};
    function y() {}
    function g() {}
    function m() {}
    var b = {};
    b[i] = function () {
      return this;
    };
    var w = Object.getPrototypeOf,
      j = w && w(w(k([])));
    j && j !== n && r.call(j, i) && (b = j);
    var x = (m.prototype = y.prototype = Object.create(b));
    function O(t) {
      ["next", "throw", "return"].forEach(function (e) {
        u(t, e, function (t) {
          return this._invoke(e, t);
        });
      });
    }
    function E(t, e) {
      function n(o, i, a, c) {
        var u = s(t[o], t, i);
        if ("throw" !== u.type) {
          var f = u.arg,
            l = f.value;
          return l && "object" == typeof l && r.call(l, "__await")
            ? e.resolve(l.__await).then(
                function (t) {
                  n("next", t, a, c);
                },
                function (t) {
                  n("throw", t, a, c);
                },
              )
            : e.resolve(l).then(
                function (t) {
                  (f.value = t), a(f);
                },
                function (t) {
                  return n("throw", t, a, c);
                },
              );
        }
        c(u.arg);
      }
      var o;
      this._invoke = function (t, r) {
        function i() {
          return new e(function (e, o) {
            n(t, r, e, o);
          });
        }
        return (o = o ? o.then(i, i) : i());
      };
    }
    function S(t, n) {
      var r = t.iterator[n.method];
      if (r === e) {
        if (((n.delegate = null), "throw" === n.method)) {
          if (
            t.iterator.return &&
            ((n.method = "return"), (n.arg = e), S(t, n), "throw" === n.method)
          )
            return d;
          (n.method = "throw"),
            (n.arg = new TypeError(
              "The iterator does not provide a 'throw' method",
            ));
        }
        return d;
      }
      var o = s(r, t.iterator, n.arg);
      if ("throw" === o.type)
        return (n.method = "throw"), (n.arg = o.arg), (n.delegate = null), d;
      var i = o.arg;
      return i
        ? i.done
          ? ((n[t.resultName] = i.value),
            (n.next = t.nextLoc),
            "return" !== n.method && ((n.method = "next"), (n.arg = e)),
            (n.delegate = null),
            d)
          : i
        : ((n.method = "throw"),
          (n.arg = new TypeError("iterator result is not an object")),
          (n.delegate = null),
          d);
    }
    function T(t) {
      var e = { tryLoc: t[0] };
      1 in t && (e.catchLoc = t[1]),
        2 in t && ((e.finallyLoc = t[2]), (e.afterLoc = t[3])),
        this.tryEntries.push(e);
    }
    function P(t) {
      var e = t.completion || {};
      (e.type = "normal"), delete e.arg, (t.completion = e);
    }
    function _(t) {
      (this.tryEntries = [{ tryLoc: "root" }]),
        t.forEach(T, this),
        this.reset(!0);
    }
    function k(t) {
      if (t) {
        var n = t[i];
        if (n) return n.call(t);
        if ("function" == typeof t.next) return t;
        if (!isNaN(t.length)) {
          var o = -1,
            a = function n() {
              for (; ++o < t.length; )
                if (r.call(t, o)) return (n.value = t[o]), (n.done = !1), n;
              return (n.value = e), (n.done = !0), n;
            };
          return (a.next = a);
        }
      }
      return { next: L };
    }
    function L() {
      return { value: e, done: !0 };
    }
    return (
      (g.prototype = x.constructor = m),
      (m.constructor = g),
      (g.displayName = u(m, c, "GeneratorFunction")),
      (t.isGeneratorFunction = function (t) {
        var e = "function" == typeof t && t.constructor;
        return (
          !!e && (e === g || "GeneratorFunction" === (e.displayName || e.name))
        );
      }),
      (t.mark = function (t) {
        return (
          Object.setPrototypeOf
            ? Object.setPrototypeOf(t, m)
            : ((t.__proto__ = m), u(t, c, "GeneratorFunction")),
          (t.prototype = Object.create(x)),
          t
        );
      }),
      (t.awrap = function (t) {
        return { __await: t };
      }),
      O(E.prototype),
      (E.prototype[a] = function () {
        return this;
      }),
      (t.AsyncIterator = E),
      (t.async = function (e, n, r, o, i) {
        void 0 === i && (i = Promise);
        var a = new E(f(e, n, r, o), i);
        return t.isGeneratorFunction(n)
          ? a
          : a.next().then(function (t) {
              return t.done ? t.value : a.next();
            });
      }),
      O(x),
      u(x, c, "Generator"),
      (x[i] = function () {
        return this;
      }),
      (x.toString = function () {
        return "[object Generator]";
      }),
      (t.keys = function (t) {
        var e = [];
        for (var n in t) e.push(n);
        return (
          e.reverse(),
          function n() {
            for (; e.length; ) {
              var r = e.pop();
              if (r in t) return (n.value = r), (n.done = !1), n;
            }
            return (n.done = !0), n;
          }
        );
      }),
      (t.values = k),
      (_.prototype = {
        constructor: _,
        reset: function (t) {
          if (
            ((this.prev = 0),
            (this.next = 0),
            (this.sent = this._sent = e),
            (this.done = !1),
            (this.delegate = null),
            (this.method = "next"),
            (this.arg = e),
            this.tryEntries.forEach(P),
            !t)
          )
            for (var n in this)
              "t" === n.charAt(0) &&
                r.call(this, n) &&
                !isNaN(+n.slice(1)) &&
                (this[n] = e);
        },
        stop: function () {
          this.done = !0;
          var t = this.tryEntries[0].completion;
          if ("throw" === t.type) throw t.arg;
          return this.rval;
        },
        dispatchException: function (t) {
          if (this.done) throw t;
          var n = this;
          function o(r, o) {
            return (
              (c.type = "throw"),
              (c.arg = t),
              (n.next = r),
              o && ((n.method = "next"), (n.arg = e)),
              !!o
            );
          }
          for (var i = this.tryEntries.length - 1; i >= 0; --i) {
            var a = this.tryEntries[i],
              c = a.completion;
            if ("root" === a.tryLoc) return o("end");
            if (a.tryLoc <= this.prev) {
              var u = r.call(a, "catchLoc"),
                f = r.call(a, "finallyLoc");
              if (u && f) {
                if (this.prev < a.catchLoc) return o(a.catchLoc, !0);
                if (this.prev < a.finallyLoc) return o(a.finallyLoc);
              } else if (u) {
                if (this.prev < a.catchLoc) return o(a.catchLoc, !0);
              } else {
                if (!f)
                  throw new Error("try statement without catch or finally");
                if (this.prev < a.finallyLoc) return o(a.finallyLoc);
              }
            }
          }
        },
        abrupt: function (t, e) {
          for (var n = this.tryEntries.length - 1; n >= 0; --n) {
            var o = this.tryEntries[n];
            if (
              o.tryLoc <= this.prev &&
              r.call(o, "finallyLoc") &&
              this.prev < o.finallyLoc
            ) {
              var i = o;
              break;
            }
          }
          i &&
            ("break" === t || "continue" === t) &&
            i.tryLoc <= e &&
            e <= i.finallyLoc &&
            (i = null);
          var a = i ? i.completion : {};
          return (
            (a.type = t),
            (a.arg = e),
            i
              ? ((this.method = "next"), (this.next = i.finallyLoc), d)
              : this.complete(a)
          );
        },
        complete: function (t, e) {
          if ("throw" === t.type) throw t.arg;
          return (
            "break" === t.type || "continue" === t.type
              ? (this.next = t.arg)
              : "return" === t.type
              ? ((this.rval = this.arg = t.arg),
                (this.method = "return"),
                (this.next = "end"))
              : "normal" === t.type && e && (this.next = e),
            d
          );
        },
        finish: function (t) {
          for (var e = this.tryEntries.length - 1; e >= 0; --e) {
            var n = this.tryEntries[e];
            if (n.finallyLoc === t)
              return this.complete(n.completion, n.afterLoc), P(n), d;
          }
        },
        catch: function (t) {
          for (var e = this.tryEntries.length - 1; e >= 0; --e) {
            var n = this.tryEntries[e];
            if (n.tryLoc === t) {
              var r = n.completion;
              if ("throw" === r.type) {
                var o = r.arg;
                P(n);
              }
              return o;
            }
          }
          throw new Error("illegal catch attempt");
        },
        delegateYield: function (t, n, r) {
          return (
            (this.delegate = { iterator: k(t), resultName: n, nextLoc: r }),
            "next" === this.method && (this.arg = e),
            d
          );
        },
      }),
      t
    );
  })(t.exports);
  try {
    regeneratorRuntime = e;
  } catch (t) {
    Function("r", "regeneratorRuntime = r")(e);
  }
})({ exports: {} });
var pn,
  hn,
  vn = Xt("navigator", "userAgent") || "",
  dn = vn,
  yn = n.process,
  gn = yn && yn.versions,
  mn = gn && gn.v8;
mn
  ? (hn = (pn = mn.split("."))[0] < 4 ? 1 : pn[0] + pn[1])
  : dn &&
    (!(pn = dn.match(/Edge\/(\d+)/)) || pn[1] >= 74) &&
    (pn = dn.match(/Chrome\/(\d+)/)) &&
    (hn = pn[1]);
var bn = hn && +hn,
  wn = bn,
  jn = o,
  xn =
    !!Object.getOwnPropertySymbols &&
    !jn(function () {
      return !String(Symbol()) || (!Symbol.sham && wn && wn < 41);
    }),
  On = xn && !Symbol.sham && "symbol" == typeof Symbol.iterator,
  En = n,
  Sn = lt.exports,
  Tn = P,
  Pn = mt,
  _n = xn,
  kn = On,
  Ln = Sn("wks"),
  Mn = En.Symbol,
  An = kn ? Mn : (Mn && Mn.withoutSetter) || Pn,
  In = function (t) {
    return (
      (Tn(Ln, t) && (_n || "string" == typeof Ln[t])) ||
        (_n && Tn(Mn, t) ? (Ln[t] = Mn[t]) : (Ln[t] = An("Symbol." + t))),
      Ln[t]
    );
  },
  Rn = {};
Rn[In("toStringTag")] = "z";
var Cn = "[object z]" === String(Rn),
  Nn = Cn,
  Fn = p,
  Jn = In("toStringTag"),
  Dn =
    "Arguments" ==
    Fn(
      (function () {
        return arguments;
      })(),
    ),
  $n = Nn
    ? Fn
    : function (t) {
        var e, n, r;
        return void 0 === t
          ? "Undefined"
          : null === t
          ? "Null"
          : "string" ==
            typeof (n = (function (t, e) {
              try {
                return t[e];
              } catch (t) {}
            })((e = Object(t)), Jn))
          ? n
          : Dn
          ? Fn(e)
          : "Object" == (r = Fn(e)) && "function" == typeof e.callee
          ? "Arguments"
          : r;
      },
  Gn = $n,
  Hn = Cn
    ? {}.toString
    : function () {
        return "[object " + Gn(this) + "]";
      },
  zn = Cn,
  Wn = Z.exports,
  qn = Hn;
zn || Wn(Object.prototype, "toString", qn, { unsafe: !0 });
var Un = n.Promise,
  Kn = Z.exports,
  Qn = w,
  Xn = W,
  Yn = function (t) {
    if (!Qn(t) && null !== t)
      throw TypeError("Can't set " + String(t) + " as a prototype");
    return t;
  },
  Bn =
    Object.setPrototypeOf ||
    ("__proto__" in {}
      ? (function () {
          var t,
            e = !1,
            n = {};
          try {
            (t = Object.getOwnPropertyDescriptor(
              Object.prototype,
              "__proto__",
            ).set).call(n, []),
              (e = n instanceof Array);
          } catch (t) {}
          return function (n, r) {
            return Xn(n), Yn(r), e ? t.call(n, r) : (n.__proto__ = r), n;
          };
        })()
      : void 0),
  Vn = H.f,
  Zn = P,
  tr = In("toStringTag"),
  er = Xt,
  nr = H,
  rr = i,
  or = In("species"),
  ir = function (t) {
    if ("function" != typeof t)
      throw TypeError(String(t) + " is not a function");
    return t;
  },
  ar = {},
  cr = ar,
  ur = In("iterator"),
  fr = Array.prototype,
  sr = ir,
  lr = function (t, e, n) {
    if ((sr(t), void 0 === e)) return t;
    switch (n) {
      case 0:
        return function () {
          return t.call(e);
        };
      case 1:
        return function (n) {
          return t.call(e, n);
        };
      case 2:
        return function (n, r) {
          return t.call(e, n, r);
        };
      case 3:
        return function (n, r, o) {
          return t.call(e, n, r, o);
        };
    }
    return function () {
      return t.apply(e, arguments);
    };
  },
  pr = $n,
  hr = ar,
  vr = In("iterator"),
  dr = W,
  yr = W,
  gr = function (t) {
    return void 0 !== t && (cr.Array === t || fr[ur] === t);
  },
  mr = ne,
  br = lr,
  wr = function (t) {
    if (null != t) return t[vr] || t["@@iterator"] || hr[pr(t)];
  },
  jr = function (t) {
    var e = t.return;
    if (void 0 !== e) return dr(e.call(t)).value;
  },
  xr = function (t, e) {
    (this.stopped = t), (this.result = e);
  },
  Or = In("iterator"),
  Er = !1;
try {
  var Sr = 0,
    Tr = {
      next: function () {
        return { done: !!Sr++ };
      },
      return: function () {
        Er = !0;
      },
    };
  (Tr[Or] = function () {
    return this;
  }),
    Array.from(Tr, function () {
      throw 2;
    });
} catch (t) {}
var Pr,
  _r,
  kr,
  Lr = W,
  Mr = ir,
  Ar = In("species"),
  Ir = Xt("document", "documentElement"),
  Rr = /(?:iphone|ipod|ipad).*applewebkit/i.test(vn),
  Cr = "process" == p(n.process),
  Nr = n,
  Fr = o,
  Jr = lr,
  Dr = Ir,
  $r = M,
  Gr = Rr,
  Hr = Cr,
  zr = Nr.location,
  Wr = Nr.setImmediate,
  qr = Nr.clearImmediate,
  Ur = Nr.process,
  Kr = Nr.MessageChannel,
  Qr = Nr.Dispatch,
  Xr = 0,
  Yr = {},
  Br = function (t) {
    if (Yr.hasOwnProperty(t)) {
      var e = Yr[t];
      delete Yr[t], e();
    }
  },
  Vr = function (t) {
    return function () {
      Br(t);
    };
  },
  Zr = function (t) {
    Br(t.data);
  },
  to = function (t) {
    Nr.postMessage(t + "", zr.protocol + "//" + zr.host);
  };
(Wr && qr) ||
  ((Wr = function (t) {
    for (var e = [], n = 1; arguments.length > n; ) e.push(arguments[n++]);
    return (
      (Yr[++Xr] = function () {
        ("function" == typeof t ? t : Function(t)).apply(void 0, e);
      }),
      Pr(Xr),
      Xr
    );
  }),
  (qr = function (t) {
    delete Yr[t];
  }),
  Hr
    ? (Pr = function (t) {
        Ur.nextTick(Vr(t));
      })
    : Qr && Qr.now
    ? (Pr = function (t) {
        Qr.now(Vr(t));
      })
    : Kr && !Gr
    ? ((kr = (_r = new Kr()).port2),
      (_r.port1.onmessage = Zr),
      (Pr = Jr(kr.postMessage, kr, 1)))
    : Nr.addEventListener &&
      "function" == typeof postMessage &&
      !Nr.importScripts &&
      zr &&
      "file:" !== zr.protocol &&
      !Fr(to)
    ? ((Pr = to), Nr.addEventListener("message", Zr, !1))
    : (Pr =
        "onreadystatechange" in $r("script")
          ? function (t) {
              Dr.appendChild($r("script")).onreadystatechange = function () {
                Dr.removeChild(this), Br(t);
              };
            }
          : function (t) {
              setTimeout(Vr(t), 0);
            }));
var eo,
  no,
  ro,
  oo,
  io,
  ao,
  co,
  uo,
  fo = { set: Wr, clear: qr },
  so = /web0s(?!.*chrome)/i.test(vn),
  lo = n,
  po = r.f,
  ho = fo.set,
  vo = Rr,
  yo = so,
  go = Cr,
  mo = lo.MutationObserver || lo.WebKitMutationObserver,
  bo = lo.document,
  wo = lo.process,
  jo = lo.Promise,
  xo = po(lo, "queueMicrotask"),
  Oo = xo && xo.value;
Oo ||
  ((eo = function () {
    var t, e;
    for (go && (t = wo.domain) && t.exit(); no; ) {
      (e = no.fn), (no = no.next);
      try {
        e();
      } catch (t) {
        throw (no ? oo() : (ro = void 0), t);
      }
    }
    (ro = void 0), t && t.enter();
  }),
  vo || go || yo || !mo || !bo
    ? jo && jo.resolve
      ? (((co = jo.resolve(void 0)).constructor = jo),
        (uo = co.then),
        (oo = function () {
          uo.call(co, eo);
        }))
      : (oo = go
          ? function () {
              wo.nextTick(eo);
            }
          : function () {
              ho.call(lo, eo);
            })
    : ((io = !0),
      (ao = bo.createTextNode("")),
      new mo(eo).observe(ao, { characterData: !0 }),
      (oo = function () {
        ao.data = io = !io;
      })));
var Eo =
    Oo ||
    function (t) {
      var e = { fn: t, next: void 0 };
      ro && (ro.next = e), no || ((no = e), oo()), (ro = e);
    },
  So = {},
  To = ir,
  Po = function (t) {
    var e, n;
    (this.promise = new t(function (t, r) {
      if (void 0 !== e || void 0 !== n)
        throw TypeError("Bad Promise constructor");
      (e = t), (n = r);
    })),
      (this.resolve = To(e)),
      (this.reject = To(n));
  };
So.f = function (t) {
  return new Po(t);
};
var _o,
  ko,
  Lo,
  Mo,
  Ao = W,
  Io = w,
  Ro = So,
  Co = n,
  No = "object" == typeof window,
  Fo = ze,
  Jo = n,
  Do = Xt,
  $o = Un,
  Go = Z.exports,
  Ho = function (t, e, n) {
    for (var r in e) Kn(t, r, e[r], n);
    return t;
  },
  zo = Bn,
  Wo = function (t, e, n) {
    t &&
      !Zn((t = n ? t : t.prototype), tr) &&
      Vn(t, tr, { configurable: !0, value: e });
  },
  qo = function (t) {
    var e = er(t),
      n = nr.f;
    rr &&
      e &&
      !e[or] &&
      n(e, or, {
        configurable: !0,
        get: function () {
          return this;
        },
      });
  },
  Uo = w,
  Ko = ir,
  Qo = function (t, e, n) {
    if (!(t instanceof e))
      throw TypeError("Incorrect " + (n ? n + " " : "") + "invocation");
    return t;
  },
  Xo = ct,
  Yo = function (t, e, n) {
    var r,
      o,
      i,
      a,
      c,
      u,
      f,
      s = n && n.that,
      l = !(!n || !n.AS_ENTRIES),
      p = !(!n || !n.IS_ITERATOR),
      h = !(!n || !n.INTERRUPTED),
      v = br(e, s, 1 + l + h),
      d = function (t) {
        return r && jr(r), new xr(!0, t);
      },
      y = function (t) {
        return l
          ? (yr(t), h ? v(t[0], t[1], d) : v(t[0], t[1]))
          : h
          ? v(t, d)
          : v(t);
      };
    if (p) r = t;
    else {
      if ("function" != typeof (o = wr(t)))
        throw TypeError("Target is not iterable");
      if (gr(o)) {
        for (i = 0, a = mr(t.length); a > i; i++)
          if ((c = y(t[i])) && c instanceof xr) return c;
        return new xr(!1);
      }
      r = o.call(t);
    }
    for (u = r.next; !(f = u.call(r)).done; ) {
      try {
        c = y(f.value);
      } catch (t) {
        throw (jr(r), t);
      }
      if ("object" == typeof c && c && c instanceof xr) return c;
    }
    return new xr(!1);
  },
  Bo = function (t, e) {
    if (!e && !Er) return !1;
    var n = !1;
    try {
      var r = {};
      (r[Or] = function () {
        return {
          next: function () {
            return { done: (n = !0) };
          },
        };
      }),
        t(r);
    } catch (t) {}
    return n;
  },
  Vo = function (t, e) {
    var n,
      r = Lr(t).constructor;
    return void 0 === r || null == (n = Lr(r)[Ar]) ? e : Mr(n);
  },
  Zo = fo.set,
  ti = Eo,
  ei = function (t, e) {
    if ((Ao(t), Io(e) && e.constructor === t)) return e;
    var n = Ro.f(t);
    return (0, n.resolve)(e), n.promise;
  },
  ni = function (t, e) {
    var n = Co.console;
    n && n.error && (1 === arguments.length ? n.error(t) : n.error(t, e));
  },
  ri = So,
  oi = function (t) {
    try {
      return { error: !1, value: t() };
    } catch (t) {
      return { error: !0, value: t };
    }
  },
  ii = Ft,
  ai = Ce,
  ci = No,
  ui = Cr,
  fi = bn,
  si = In("species"),
  li = "Promise",
  pi = ii.get,
  hi = ii.set,
  vi = ii.getterFor(li),
  di = $o && $o.prototype,
  yi = $o,
  gi = di,
  mi = Jo.TypeError,
  bi = Jo.document,
  wi = Jo.process,
  ji = ri.f,
  xi = ji,
  Oi = !!(bi && bi.createEvent && Jo.dispatchEvent),
  Ei = "function" == typeof PromiseRejectionEvent,
  Si = !1,
  Ti = ai(li, function () {
    var t = Xo(yi) !== String(yi);
    if (!t && 66 === fi) return !0;
    if (fi >= 51 && /native code/.test(yi)) return !1;
    var e = new yi(function (t) {
        t(1);
      }),
      n = function (t) {
        t(
          function () {},
          function () {},
        );
      };
    return (
      ((e.constructor = {})[si] = n),
      !(Si = e.then(function () {}) instanceof n) || (!t && ci && !Ei)
    );
  }),
  Pi =
    Ti ||
    !Bo(function (t) {
      yi.all(t).catch(function () {});
    }),
  _i = function (t) {
    var e;
    return !(!Uo(t) || "function" != typeof (e = t.then)) && e;
  },
  ki = function (t, e) {
    if (!t.notified) {
      t.notified = !0;
      var n = t.reactions;
      ti(function () {
        for (var r = t.value, o = 1 == t.state, i = 0; n.length > i; ) {
          var a,
            c,
            u,
            f = n[i++],
            s = o ? f.ok : f.fail,
            l = f.resolve,
            p = f.reject,
            h = f.domain;
          try {
            s
              ? (o || (2 === t.rejection && Ii(t), (t.rejection = 1)),
                !0 === s
                  ? (a = r)
                  : (h && h.enter(), (a = s(r)), h && (h.exit(), (u = !0))),
                a === f.promise
                  ? p(mi("Promise-chain cycle"))
                  : (c = _i(a))
                  ? c.call(a, l, p)
                  : l(a))
              : p(r);
          } catch (t) {
            h && !u && h.exit(), p(t);
          }
        }
        (t.reactions = []), (t.notified = !1), e && !t.rejection && Mi(t);
      });
    }
  },
  Li = function (t, e, n) {
    var r, o;
    Oi
      ? (((r = bi.createEvent("Event")).promise = e),
        (r.reason = n),
        r.initEvent(t, !1, !0),
        Jo.dispatchEvent(r))
      : (r = { promise: e, reason: n }),
      !Ei && (o = Jo["on" + t])
        ? o(r)
        : "unhandledrejection" === t && ni("Unhandled promise rejection", n);
  },
  Mi = function (t) {
    Zo.call(Jo, function () {
      var e,
        n = t.facade,
        r = t.value;
      if (
        Ai(t) &&
        ((e = oi(function () {
          ui
            ? wi.emit("unhandledRejection", r, n)
            : Li("unhandledrejection", n, r);
        })),
        (t.rejection = ui || Ai(t) ? 2 : 1),
        e.error)
      )
        throw e.value;
    });
  },
  Ai = function (t) {
    return 1 !== t.rejection && !t.parent;
  },
  Ii = function (t) {
    Zo.call(Jo, function () {
      var e = t.facade;
      ui ? wi.emit("rejectionHandled", e) : Li("rejectionhandled", e, t.value);
    });
  },
  Ri = function (t, e, n) {
    return function (r) {
      t(e, r, n);
    };
  },
  Ci = function (t, e, n) {
    t.done ||
      ((t.done = !0), n && (t = n), (t.value = e), (t.state = 2), ki(t, !0));
  },
  Ni = function (t, e, n) {
    if (!t.done) {
      (t.done = !0), n && (t = n);
      try {
        if (t.facade === e) throw mi("Promise can't be resolved itself");
        var r = _i(e);
        r
          ? ti(function () {
              var n = { done: !1 };
              try {
                r.call(e, Ri(Ni, n, t), Ri(Ci, n, t));
              } catch (e) {
                Ci(n, e, t);
              }
            })
          : ((t.value = e), (t.state = 1), ki(t, !1));
      } catch (e) {
        Ci({ done: !1 }, e, t);
      }
    }
  };
if (
  Ti &&
  ((gi = (yi = function (t) {
    Qo(this, yi, li), Ko(t), _o.call(this);
    var e = pi(this);
    try {
      t(Ri(Ni, e), Ri(Ci, e));
    } catch (t) {
      Ci(e, t);
    }
  }).prototype),
  ((_o = function (t) {
    hi(this, {
      type: li,
      done: !1,
      notified: !1,
      parent: !1,
      reactions: [],
      rejection: !1,
      state: 0,
      value: void 0,
    });
  }).prototype = Ho(gi, {
    then: function (t, e) {
      var n = vi(this),
        r = ji(Vo(this, yi));
      return (
        (r.ok = "function" != typeof t || t),
        (r.fail = "function" == typeof e && e),
        (r.domain = ui ? wi.domain : void 0),
        (n.parent = !0),
        n.reactions.push(r),
        0 != n.state && ki(n, !1),
        r.promise
      );
    },
    catch: function (t) {
      return this.then(void 0, t);
    },
  })),
  (ko = function () {
    var t = new _o(),
      e = pi(t);
    (this.promise = t), (this.resolve = Ri(Ni, e)), (this.reject = Ri(Ci, e));
  }),
  (ri.f = ji =
    function (t) {
      return t === yi || t === Lo ? new ko(t) : xi(t);
    }),
  "function" == typeof $o && di !== Object.prototype)
) {
  (Mo = di.then),
    Si ||
      (Go(
        di,
        "then",
        function (t, e) {
          var n = this;
          return new yi(function (t, e) {
            Mo.call(n, t, e);
          }).then(t, e);
        },
        { unsafe: !0 },
      ),
      Go(di, "catch", gi.catch, { unsafe: !0 }));
  try {
    delete di.constructor;
  } catch (t) {}
  zo && zo(di, gi);
}
Fo({ global: !0, wrap: !0, forced: Ti }, { Promise: yi }),
  Wo(yi, li, !1),
  qo(li),
  (Lo = Do(li)),
  Fo(
    { target: li, stat: !0, forced: Ti },
    {
      reject: function (t) {
        var e = ji(this);
        return e.reject.call(void 0, t), e.promise;
      },
    },
  ),
  Fo(
    { target: li, stat: !0, forced: Ti },
    {
      resolve: function (t) {
        return ei(this, t);
      },
    },
  ),
  Fo(
    { target: li, stat: !0, forced: Pi },
    {
      all: function (t) {
        var e = this,
          n = ji(e),
          r = n.resolve,
          o = n.reject,
          i = oi(function () {
            var n = Ko(e.resolve),
              i = [],
              a = 0,
              c = 1;
            Yo(t, function (t) {
              var u = a++,
                f = !1;
              i.push(void 0),
                c++,
                n.call(e, t).then(function (t) {
                  f || ((f = !0), (i[u] = t), --c || r(i));
                }, o);
            }),
              --c || r(i);
          });
        return i.error && o(i.value), n.promise;
      },
      race: function (t) {
        var e = this,
          n = ji(e),
          r = n.reject,
          o = oi(function () {
            var o = Ko(e.resolve);
            Yo(t, function (t) {
              o.call(e, t).then(n.resolve, r);
            });
          });
        return o.error && r(o.value), n.promise;
      },
    },
  );
var Fi,
  Ji = H,
  Di = W,
  $i = Ue,
  Gi = i
    ? Object.defineProperties
    : function (t, e) {
        Di(t);
        for (var n, r = $i(e), o = r.length, i = 0; o > i; )
          Ji.f(t, (n = r[i++]), e[n]);
        return t;
      },
  Hi = W,
  zi = Gi,
  Wi = ye,
  qi = Ot,
  Ui = Ir,
  Ki = M,
  Qi = xt("IE_PROTO"),
  Xi = function () {},
  Yi = function (t) {
    return "<script>" + t + "</script>";
  },
  Bi = function () {
    try {
      Fi = document.domain && new ActiveXObject("htmlfile");
    } catch (t) {}
    var t, e;
    Bi = Fi
      ? (function (t) {
          t.write(Yi("")), t.close();
          var e = t.parentWindow.Object;
          return (t = null), e;
        })(Fi)
      : (((e = Ki("iframe")).style.display = "none"),
        Ui.appendChild(e),
        (e.src = String("javascript:")),
        (t = e.contentWindow.document).open(),
        t.write(Yi("document.F=Object")),
        t.close(),
        t.F);
    for (var n = Wi.length; n--; ) delete Bi.prototype[Wi[n]];
    return Bi();
  };
qi[Qi] = !0;
var Vi =
    Object.create ||
    function (t, e) {
      var n;
      return (
        null !== t
          ? ((Xi.prototype = Hi(t)),
            (n = new Xi()),
            (Xi.prototype = null),
            (n[Qi] = t))
          : (n = Bi()),
        void 0 === e ? n : zi(n, e)
      );
    },
  Zi = H,
  ta = In("unscopables"),
  ea = Array.prototype;
null == ea[ta] && Zi.f(ea, ta, { configurable: !0, value: Vi(null) });
var na = se.includes,
  ra = function (t) {
    ea[ta][t] = !0;
  };
ze(
  { target: "Array", proto: !0 },
  {
    includes: function (t) {
      return na(this, t, arguments.length > 1 ? arguments[1] : void 0);
    },
  },
),
  ra("includes");
var oa = w,
  ia = p,
  aa = In("match"),
  ca = function (t) {
    var e;
    return oa(t) && (void 0 !== (e = t[aa]) ? !!e : "RegExp" == ia(t));
  },
  ua = In("match"),
  fa = function (t) {
    if (ca(t)) throw TypeError("The method doesn't accept regular expressions");
    return t;
  },
  sa = y;
ze(
  {
    target: "String",
    proto: !0,
    forced: !(function (t) {
      var e = /./;
      try {
        "/./"[t](e);
      } catch (n) {
        try {
          return (e[ua] = !1), "/./"[t](e);
        } catch (t) {}
      }
      return !1;
    })("includes"),
  },
  {
    includes: function (t) {
      return !!~String(sa(this)).indexOf(
        fa(t),
        arguments.length > 1 ? arguments[1] : void 0,
      );
    },
  },
);
var la = function () {
    var t,
      e = {
        messageStyle: "none",
        tex2jax: {
          inlineMath: [
            ["$", "$"],
            ["\\(", "\\)"],
          ],
          skipTags: ["script", "noscript", "style", "textarea", "pre"],
        },
        skipStartupTypeset: !0,
      };
    return {
      id: "mathjax2",
      init: function (n) {
        var r = (t = n).getConfig().mathjax2 || t.getConfig().math || {},
          o = on(on({}, e), r),
          i =
            (o.mathjax || "https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js") +
            "?config=" +
            (o.config || "TeX-AMS_HTML-full");
        (o.tex2jax = on(on({}, e.tex2jax), r.tex2jax)),
          (o.mathjax = o.config = null),
          (function (t, e) {
            var n = this,
              r = document.querySelector("head"),
              o = document.createElement("script");
            (o.type = "text/javascript"), (o.src = t);
            var i = function () {
              "function" == typeof e && (e.call(), (e = null));
            };
            (o.onload = i),
              (o.onreadystatechange = function () {
                "loaded" === n.readyState && i();
              }),
              r.appendChild(o);
          })(i, function () {
            MathJax.Hub.Config(o),
              MathJax.Hub.Queue(["Typeset", MathJax.Hub, t.getRevealElement()]),
              MathJax.Hub.Queue(t.layout),
              t.on("slidechanged", function (t) {
                MathJax.Hub.Queue(["Typeset", MathJax.Hub, t.currentSlide]);
              });
          });
      },
    };
  },
  pa = la,
  ha = (Plugin = Object.assign(pa(), {
    KaTeX: function () {
      var t,
        e = {
          version: "latest",
          delimiters: [
            { left: "$$", right: "$$", display: !0 },
            { left: "$", right: "$", display: !1 },
            { left: "\\(", right: "\\)", display: !1 },
            { left: "\\[", right: "\\]", display: !0 },
          ],
          ignoredTags: ["script", "noscript", "style", "textarea", "pre"],
        },
        n = function (t) {
          return new Promise(function (e, n) {
            var r = document.createElement("script");
            (r.type = "text/javascript"),
              (r.onload = e),
              (r.onerror = n),
              (r.src = t),
              document.head.append(r);
          });
        };
      function r() {
        return (r = cn(
          regeneratorRuntime.mark(function t(e) {
            var r, o, i;
            return regeneratorRuntime.wrap(
              function (t) {
                for (;;)
                  switch ((t.prev = t.next)) {
                    case 0:
                      (r = ln(e)), (t.prev = 1), r.s();
                    case 3:
                      if ((o = r.n()).done) {
                        t.next = 9;
                        break;
                      }
                      return (i = o.value), (t.next = 7), n(i);
                    case 7:
                      t.next = 3;
                      break;
                    case 9:
                      t.next = 14;
                      break;
                    case 11:
                      (t.prev = 11), (t.t0 = t.catch(1)), r.e(t.t0);
                    case 14:
                      return (t.prev = 14), r.f(), t.finish(14);
                    case 17:
                    case "end":
                      return t.stop();
                  }
              },
              t,
              null,
              [[1, 11, 14, 17]],
            );
          }),
        )).apply(this, arguments);
      }
      return {
        id: "katex",
        init: function (n) {
          var o = this,
            i = (t = n).getConfig().katex || {},
            a = on(on({}, e), i);
          a.local, a.version, a.extensions;
          var c = fn(a, ["local", "version", "extensions"]),
            u = a.local || "https://cdn.jsdelivr.net/npm/katex",
            f = a.local ? "" : "@" + a.version,
            s = u + f + "/dist/katex.min.css",
            l = u + f + "/dist/contrib/mhchem.min.js",
            p = u + f + "/dist/contrib/auto-render.min.js",
            h = [u + f + "/dist/katex.min.js"];
          a.extensions && a.extensions.includes("mhchem") && h.push(l),
            h.push(p);
          var v,
            d,
            y = function () {
              renderMathInElement(n.getSlidesElement(), c), t.layout();
            };
          (v = s),
            ((d = document.createElement("link")).rel = "stylesheet"),
            (d.href = v),
            document.head.appendChild(d),
            (function (t) {
              return r.apply(this, arguments);
            })(h).then(function () {
              t.isReady() ? y() : t.on("ready", y.bind(o));
            });
        },
      };
    },
    MathJax2: la,
    MathJax3: function () {
      var t = {
        tex: {
          inlineMath: [
            ["$", "$"],
            ["\\(", "\\)"],
          ],
        },
        options: {
          skipHtmlTags: ["script", "noscript", "style", "textarea", "pre"],
        },
        startup: {
          ready: function () {
            MathJax.startup.defaultReady(),
              MathJax.startup.promise.then(function () {
                Reveal.layout();
              });
          },
        },
      };
      return {
        id: "mathjax3",
        init: function (e) {
          var n = e.getConfig().mathjax3 || {},
            r = on(on({}, t), n);
          (r.tex = on(on({}, t.tex), n.tex)),
            (r.options = on(on({}, t.options), n.options)),
            (r.startup = on(on({}, t.startup), n.startup));
          var o =
            r.mathjax ||
            "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js";
          (r.mathjax = null),
            (window.MathJax = r),
            (function (t, e) {
              var n = document.createElement("script");
              (n.type = "text/javascript"),
                (n.id = "MathJax-script"),
                (n.src = t),
                (n.async = !0),
                (n.onload = function () {
                  "function" == typeof e && (e.call(), (e = null));
                }),
                document.head.appendChild(n);
            })(o, function () {
              Reveal.addEventListener("slidechanged", function (t) {
                MathJax.typeset();
              });
            });
        },
      };
    },
  }));
export default ha;
