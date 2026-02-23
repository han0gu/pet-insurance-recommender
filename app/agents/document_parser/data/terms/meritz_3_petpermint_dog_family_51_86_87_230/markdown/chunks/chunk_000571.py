from langchain_core.documents import Document

chunk = Document(
    page_content=('| 5 | 피부질환 | AFA007 | 비만세포종 (피부) (양성) |\n'
 '| 5 | 피부질환 | AFB007 | 악성 비만세포종 (피부) (악성) 비만세포종(피부) (양성 또는 악성이 불 확실한) |\n'
 '| 5 | 피부질환 | AFC007 | 흑색종 (양성) |\n'
 '| 5 | 피부질환 | AFA008 | 흑색종 (악성) |\n'
 '| 5 | 피부질환 | AFB008 AFC008 | 흑색종 (양성 또는 악성이 불확실한) |\n'
 '| 5 | 피부질환 | AFB009 | 피부 림프종 |\n'
 '| 5 | 피부질환 | AFB010 | 편평세포암종 |'),
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
