from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이하 의료비<br>라 합니다)에서 반려동물의료비보험금 및 보험증권에 기재된 자기부담금을 제외<br>한 금액을 제3항에 따라 '
 "보험수익자에게 주요치료보험금으로 지급합니다.<br>< 보험가입금액 1,000만원 기준 ></p><br><table id='197' "
 'style=\'font-size:14px\'><thead><tr><td colspan="2">치 료 구 '
 '분</td><td>지급한도</td><td>치료구분별 보상한도액</td></tr></thead><tbody><tr><td'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'deductible', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000992',
              'chunk_char_len': 266,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
