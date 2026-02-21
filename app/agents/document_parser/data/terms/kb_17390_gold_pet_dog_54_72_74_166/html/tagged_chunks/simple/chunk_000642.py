from langchain_core.documents import Document

chunk = Document(
    page_content=('진단확정된<br>경우에는 아래에 정한 금액을 최초 1회에 한하여 6대호흡계특정질환진단비로 보험수<br>익자에게 '
 "지급합니다.</p><br><table id='179' style='font-size:16px'><thead><tr><td "
 'rowspan="2">구 분</td><td colspan="2">지 급 금 액</td></tr><tr><td>보험계약일부터 '
 '1년미만</td><td>보험계약일부터 '
 '1년이상</td></tr></thead><tbody><tr><td>6대호흡계특정질환진단비</td><td>이 특별약관의 보험가입금액'),
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
 'indexing': {'chunk_id': 'chunk_000642',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
