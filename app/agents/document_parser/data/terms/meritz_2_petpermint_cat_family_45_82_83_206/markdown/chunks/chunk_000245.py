from langchain_core.documents import Document

chunk = Document(
    page_content=('- 요, 기타 이들과 유사한 사태\n'
 '- ④ 핵연료물질 또는 핵연료물질에 의하여 오염된 물질의\n'
 '- 방사성, 폭발성, 그 밖의 유해한 특성 또는 이들의\n'
 '- 특성에 의한 사고\n'
 '- ⑤ 제4호 이외의 방사선을 쬐는 것 또는 방사능 오염\n'
 '# 【핵연료물질】# 사용된 연료를 포함합니다.# 【핵연료물질에 의하여 오염된 물질】# 원자핵분열 생성물을 포함합니다.- ⑥ 최초 계약의 '
 '보험계약일 이전에 이미 감염 또는 발병\n'
 '- 한 질병 및 상해\n'
 '- ⑦ 원인이 어떠한 경우에도 반려동물에 대한 사료제공 또\n'
 '- 는 급수 등 기본적인 관리에 대한 태만'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
