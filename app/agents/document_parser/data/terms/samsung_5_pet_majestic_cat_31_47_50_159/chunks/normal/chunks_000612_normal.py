from langchain_core.documents import Document

chunk = Document(
    page_content=('기 간이 끝나는 날의 다음날에 특별약관이 해지된다는 내용\n'
 '3. 계약자가 회사로부터 보험계약대출을 받은 경우 특별약관이 해지되는 즉시 해약환 급금에서 보험계약대출원금과 이자가 차감된다는 내용'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 104},
 'term_type': 'special',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000612',
              'chunk_char_len': 105,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
