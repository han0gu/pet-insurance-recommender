from langchain_core.documents import Document

chunk = Document(
    page_content=('- 상\n'
 '- 인이 아닌 경우에는 본인의 인감증명서, 본인서명사실확인서 또는 안전성과 해\n'
 '- 신뢰성이 확보된 전자적 수단을 활용한 보험수익자 의사표시의 확인방법 포 및\n'
 '- 함) 질\n'
 '- 7. 기타 보험수익자가 보험금의 수령에 필요하여 제출하는 서류 병\n'
 '\uf000 제1항 제4호의 사고증명서는 수의사법 제2조(정의)에서 규정한 동물병원에서 수의\n'
 '사가 발급한 것이어야 합니다.# 관 련 법 규 수의사법 제2조(정의)# 이 법에서 사용하는 용어의 뜻은 다음과 같다.- 1. "수의사"란 '
 '수의업무를 담당하는 사람으로서 농림축산식품부장관의 면허를\n'
 '- 물'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000613',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
