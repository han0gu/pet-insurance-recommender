from langchain_core.documents import Document

chunk = Document(
    page_content=('제15조 (사기에 의한 계약)\n'
 '계약자 또는 피보험자의 사기에 의하여 계약이 성립되었음을 회사가 증명하는 경우에는 계약체결일부터 5년 이내(사기사실을 안 날부터 1개월 '
 '이내)에 계약을 취소할 수 있습니 다.\n'
 '제16조 (특별약관의 무효)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 102},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000596',
              'chunk_char_len': 130,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
