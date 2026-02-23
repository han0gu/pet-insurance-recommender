from langchain_core.documents import Document

chunk = Document(
    page_content=('제1조(보험금의 지급사유)\n'
 '약\n'
 '\uf000 회사는 피보험자가 이 특별약관의 보험기간 중에 상해의 직접결과로써 "골절진단\n'
 '관\n'
 '(치아파절제외)"로 진단확정 되고 그 치료를 목적으로 체내에 삽입한 철심을 제\n'
 '거하는 "골절철심제거술"을 받은 경우 이 특별약관의 보험가입금액을 연간 1회에\n'
 '한하여 보험수익자에게 지급합니다.\n'
 '\uf000 제1항에서 "연간"이란 계약일로부터 매1년 단위로 도래하는 계약해당일 전까지- \n'
 '기간을 의미합니다.제2조(보험금 지급에 관한 세부규정)\n'
 '보험수익자와 회사가 제1조(보험금의 지급사유)의 보험금 지급사유에 대해 합의하지'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000330',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
