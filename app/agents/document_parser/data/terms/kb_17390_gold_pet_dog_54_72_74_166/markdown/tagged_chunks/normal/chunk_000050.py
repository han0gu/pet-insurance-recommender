from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 제1항에 따라 지정된 보험수익자가 보험기간 중에 사망한 때에는 계약자는 다시 보\n'
 '험수익자를 지정할 수 있으며, 이 경우에 계약자가 보험수익자를 지정하지 않고 사\n'
 '망한때에는 보험수익자의 법정상속인을 보험수익자로 합니다.# 용 어 풀 이 법정상속인| 피상속인의 사망으로 | 인하여 민법에서 정한 '
 '상속순서에 따라 상속이 되는자를 |\n'
 '| --- | --- |\n'
 '말합니다.\n'
 '※ 상속순위# ① 피상속인의직계비속 ② 피상속인의 직계존속③ 피상속인의 형제자매 ④ 피상속인의 4촌 이내의 방계혈족-'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000050',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
