from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 지정대리청구인 변경신청서(회사양식) 2. 지정대리청구인의 주민등록등본, 가족관계등록부(기본증명서 등) 3. 신분증(주민등록증이나 '
 '운전면허증 등 사진이 붙은 정부기관발행 신분증, 본인이 아 닌 경우에는 본인의 인감증명서, 본인서명사실확인서 또는 안전성과 신뢰성이 '
 '확보된 전자적 수단을 활용한 피보험자 의사표시의 확인방법 포함)\n'
 '② 회사는 제1항의 지정대리청구인 변경 지정시 계약자의 지정 편의를 위해 가족관계서류 의 수령을 생략할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 42},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000230',
              'chunk_char_len': 250,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
