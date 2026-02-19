from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 납입최고(독촉)기간 내에 연체보험료를 납입하여야 한다는 내용 2. 납입최고(독촉)기간이 끝나는 날까지 보험료를 납입하지 않을 경우 '
 '납입최고(독촉) 기간이 끝나는 날의 다음날에 계약이 해지된다는 내용 3. 계약자가 회사로부터 보험계약대출을 받은 경우 계약이 해지되는 '
 '즉시 해약환급금 에서 보험계약대출원금과 이자가 차감된다는 내용'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 55},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000256',
              'chunk_char_len': 185,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
