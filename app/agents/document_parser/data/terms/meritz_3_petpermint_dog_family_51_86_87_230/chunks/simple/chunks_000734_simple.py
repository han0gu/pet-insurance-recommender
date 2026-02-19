from langchain_core.documents import Document

chunk = Document(
    page_content=('장해의 대상이 되지 않으나, 선천적으로 영구치 결 손이 있는 경우에는 유치의 결손을 후유장해로 평가 한다.\n'
 '16) 가철성 보철물(신체의 일부에 붙였다 떼었다 할 수 있는 틀니 등)의 파손은 후유장해의 대상이 되지 않 는다.\n'
 '5. 외모의 추상(추한 모습)장해\n'
 '가. 장해의 분류\n'
 '장해의 분류 | 지급률\n'
 '1) 외모에 뚜렷한 추상(추한 모습)을 남긴 때 | 15\n'
 '2) 외모에 약간의 추상(추한 모습)을 남긴 때 | 5\n'
 '나. 장해판정기준'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 209},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['head', 'skin', 'other']},
 'indexing': {'chunk_id': 'chunk_000734',
              'chunk_char_len': 239,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
