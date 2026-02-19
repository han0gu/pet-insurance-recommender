from langchain_core.documents import Document

chunk = Document(
    page_content=('진단계약 | 계약을 체결하기 위하여 반려동물이 건강진단 을 받아야 하는 계약을 말합니다.\n'
 '피보험자 | 반려동물의 소유와 관련하여 보험사고로 손해 를 입은 사람(법인인 경우에는 그 이사 또는 법인의 업무를 집행하는 그 밖의 '
 '기관)을 말하 며, 보험증권에 기재된 반려동물의 소유자에 한합니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 89},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000182',
              'chunk_char_len': 161,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
