from langchain_core.documents import Document

chunk = Document(
    page_content=('제19조 (사기에 의한 계약)\n'
 '계약자 또는 피보험자가 대리진단, 약물사용을 수단으로 진단절차를 통과하거나 진단서 위·변조 또는 청약일 이전에 암 또는 '
 '사람면역결핍바이러스(HIV) 감염의 진단확정을 받 은 후 이를 숨기고 가입하는 등 사기에 의하여 계약이 성립되었음을 회사가 증명하는 경 '
 '우에는 계약일부터 5년 이내(사기사실을 안 날부터 1개월 이내)에 계약을 취소할 수 있습 니다.\n'
 '제4관 보험계약의 성립과 유지\n'
 '제 20조 (보험계약의 성립)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 38},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000081',
              'chunk_char_len': 247,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
