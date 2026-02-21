from langchain_core.documents import Document

chunk = Document(
    page_content=('이외의 보험금은 피보험자로 합니다.<br>\uf000 제1항에 따라 지정된 보험수익자가 보험기간 중에 사망한 때에는 계약자는 다시 '
 '보<br>험수익자를 지정할 수 있으며, 이 경우에 계약자가 보험수익자를 지정하지 않고 사<br>망한때에는 보험수익자의 법정상속인을 '
 "보험수익자로 합니다.</p><br><h1 id='95' style='font-size:14px'>용 어 풀 이 "
 "법정상속인</h1><br><table id='96' "
 "style='font-size:14px'><thead></thead><tbody><tr><td>피상속인의"),
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
 'indexing': {'chunk_id': 'chunk_000080',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
