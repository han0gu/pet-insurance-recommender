from langchain_core.documents import Document

chunk = Document(
    page_content=("오염된 물질】<br>원자핵분열 생성물을 포함합니다.</p><br><p id='47' data-category='paragraph' "
 "style='font-size:20px'>⑥ 최초 계약의 보험계약일 이전에 이미 감염 또는 발병<br>한 질병 및 "
 "상해</p><footer id='48' style='font-size:14px'>161</footer><p id='49' "
 "data-category='list' style='font-size:16px'>⑦ 원인이 어떠한 경우에도 반려동물에 대한 사료제공 "
 '또<br>는 급수 등 기본적인 관리에'),
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
