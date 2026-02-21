from langchain_core.documents import Document

chunk = Document(
    page_content=(". 치아파절진단비(연간3회한)</h1><br><p id='231' data-category='list' "
 "style='font-size:14px'>제1조(보험금의 지급사유)<br>\uf000 회사는 피보험자가 이 특별약관의 보험기간 중에 "
 '상해의 직접결과로써【별표<br>5】치아파절 분류표에서 정한 치아파절로 진단확정 된 경우 연간 3회에 한하여<br>사고시마다 이 특별약관의 '
 '보험가입금액을 치아파절진단비로 보험수익자에게<br>지급합니다.<br>\uf000 제1항에서 "연간"이란 계약일로부터 매1년 단위로 '
 '도래하는 계약해당일 전일까<br>지 기간을'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000502',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
