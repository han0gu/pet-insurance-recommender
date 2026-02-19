from langchain_core.documents import Document

chunk = Document(
    page_content=('이 특별약관에서 정하지 않은 사항은 보험계약을 따릅니다.\n'
 '- 132 -\n'
 '별표 및 참고 별표 약관에서 인용된 법·규정 특별약관 색인\n'
 '별표\n'
 '구분 | 기간 | 지급이자\n'
 '보장관련 보험금 | 지급기일의 다음날부터 30일이내 기간 | 보험계약대출이율\n'
 '지급기일의 31일 이후부터 60일 이내 기간 | 보험계약대출이율 +가산이율(4.0%)\n'
 '지급기일의 61일 이후부터 90일 이내 기간 | 보험계약대출이율 +가산이율(6.0%)\n'
 '지급기일의 91일 이후 기간 | 보험계약대출이율 +가산이율(8.0%)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 135},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000857',
              'chunk_char_len': 268,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
