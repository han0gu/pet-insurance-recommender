from langchain_core.documents import Document

chunk = Document(
    page_content=('의 경우는 피보험자로 합니다.② 제1항에 따라 지정된 보험수익자가 보험기간 중에 사망한 때에는 계약자는 다시 보험\n'
 '수익자를 지정할 수 있으며, 이 경우에 계약자가 보험수익자를 지정하지 않고 사망한\n'
 '때에는 보험수익자의 법정상속인을 보험수익자로 합니다.<용어풀이># [법정상속인]피상속인의 사망에 의하여 민법의 규정에 의한 상속순위에 '
 '따라 상속받는 자를 말합니다.# ※ 상속순위| ① 피상속인의 직계비속 | ② 피상속인의 직계존속 |\n'
 '| --- | --- |\n'
 '| ③ 피상속인의 형제자매 | ④ 피상속인의 4촌 이내의 방계혈족 |'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000048',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
