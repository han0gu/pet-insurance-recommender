from langchain_core.documents import Document

chunk = Document(
    page_content=('편물에 대한 기록이 남는 방법으로 회사가 알린 사항은 일반적으로 도달에 필요한\n'
 '기간이 지난 때에 계약자 또는 보험수익자에게 도달된 것으로 봅니다.| 제12조(보험수익자의 | 지정) |\n'
 '| --- | --- |\n'
 '\uf000 보험수익자를 지정하지 않은 때에는 보험수익자를 제9조(만기환급금의 지급) 제1\n'
 '항의 경우는 계약자로 하고, 사망보험금의 경우는 피보험자의 법정상속인으로 하\n'
 '며, 이외의 보험금은 피보험자로 합니다.\n'
 '\uf000 제1항에 따라 지정된 보험수익자가 보험기간 중에 사망한 때에는 계약자는 다시 보'),
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
 'indexing': {'chunk_id': 'chunk_000049',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
