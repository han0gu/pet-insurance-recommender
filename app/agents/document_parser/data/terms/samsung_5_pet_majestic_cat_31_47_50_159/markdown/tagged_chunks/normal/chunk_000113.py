from langchain_core.documents import Document

chunk = Document(
    page_content=('- 1. 납입최고(독촉)기간 내에 연체보험료를 납입하여야 한다는 내용\n'
 '- 2. 납입최고(독촉)기간이 끝나는 날까지 보험료를 납입하지 않을 경우 납입최고(독촉)기\n'
 '- 간이 끝나는 날의 다음날에 계약이 해지된다는 내용\n'
 '- 3. 계약자가 회사로부터 보험계약대출을 받은 경우 계약이 해지되는 즉시 해약환급금에\n'
 '- 서 보험계약대출원금과 이자가 차감된다는 내용\n'
 '- ② 납입최고(독촉)기간의 마지막 날이 영업일이 아닌 때에는 최고(독촉)기간은 그 다음\n'
 '- 날까지로 합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000113',
              'chunk_char_len': 257,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
