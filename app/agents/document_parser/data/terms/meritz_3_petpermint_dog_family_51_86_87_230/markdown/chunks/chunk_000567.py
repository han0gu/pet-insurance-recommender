from langchain_core.documents import Document

chunk = Document(
    page_content=('| 4 | 비뇨기과 질환 | AGA001 | 신장의 양성 신생물 신장의 악성 신생물 |\n'
 '| 4 | 비뇨기과 질환 | AGB001 |  |\n'
 '| 4 | 비뇨기과 질환 | AGC001 | 신장의 신생물 (양성 또는 악성이 불확실 한) |\n'
 '196| 구 분 | 특정질병 | 분류코드 | 항목명 |\n'
 '| --- | --- | --- | --- |\n'
 '|  |  | AGB002 | 이행상피세포암종 (방광) |\n'
 '| AGA003 | 기타 방광의 양성 신생물 |  |  |\n'
 '| AGB003 | 기타 방광의 악성 신생물 |  |  |'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
