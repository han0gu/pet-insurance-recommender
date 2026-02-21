from langchain_core.documents import Document

chunk = Document(
    page_content=('. 계약자, 피보험자가 동일하고 보험수익자가 계약자의 법정상속인인 계약일 경우<br>\uf000 제3항에 따라 계약이 취소된 경우에는 '
 '회사는 이미 납입한 보험료를 계약자에게<br>돌려 드리며, 보험료를 받은 기간에 대하여 보험계약대출이율을 연단위 복리로 계 '
 "보</p><br><table id='216' "
 "style='font-size:18px'><thead></thead><tbody><tr><td>산한 금액을 "
 '더하여</td><td>지급합니다.</td><td>통약</td></tr><tr><td colspan="2">용 어 풀 이'),
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
 'indexing': {'chunk_id': 'chunk_000174',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
