from langchain_core.documents import Document

chunk = Document(
    page_content=('- 생겼음을 알았을 때\n'
 '- ② 이 계약에서 보장하는 위험과 동일한 위험을 보장하는\n'
 '- 계약을 다른 보험자와 체결하고자 할 때 또는 이와 같\n'
 '- 은 계약이 있음을 알았을 때\n'
 '- ③ 반려동물을 양도할 때\n'
 '- ④ 위 이외에 위험이 뚜렷이 변경되거나 변경되었음을 알\n'
 '- 았을 때\n'
 '\uf000 회사는 제1항의 통지로 인하여 위험의 변동이 발생한 경\n'
 '우에는 제13조(계약내용의 변경 등)에 따라 계약내용을 변\n'
 '경할 수 있습니다.91![image](/image/placeholder)\n'
 '[위험변경에 따른 계약변경 절차]\n'
 '위험변경사항 통지'),
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
