from langchain_core.documents import Document

chunk = Document(
    page_content='. 진단서 상의 분류번호는 진단 당시 시행되고 있는 한국표준질병사인분류 질병코딩지침서에 따라 기재된 것을 인정합니다. |',
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
