from langchain_core.documents import Document

chunk = Document(
    page_content=('- 2. 납입최고(독촉)기간이 끝나는 날까지 보험료를 납입하지 않을 경우 납입최고\n'
 '- (독촉)기간이 끝나는 날의 다음날에 특별약관이 해지된다는 내용\n'
 '- 3. 계약자가 회사로부터 보험계약대출을 받은 경우 특별약관이 해지되는 즉시 해\n'
 '- 약환급금에서 보험계약대출원금과 이자가 차감된다는 내용\n'
 '- \uf000 납입최고(독촉)기간의 마지막 날이 영업일이 아닌 때에는 최고(독촉)기간은 그\n'
 '- 다음 날까지로 합니다.\n'
 '- \uf000 보험수익자와 계약자가 다른 경우 보험수익자에게도 제1항에 따른 내용을 알려드\n'
 '- 립니다.'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000514',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
