from langchain_core.documents import Document

chunk = Document(
    page_content=('- 승낙하며, 특별약관은 청약한 때부터 특별부활(효력회복) 됩니다.\n'
 '- ③ 회사는 제1항의 통지를 지정된 보험수익자에게 하여야 합니다. 다만, 회사는 법정상속\n'
 '- 인이 보험수익자로 지정된 경우에는 제1항의 통지를 계약자에게 할 수 있습니다.\n'
 '# <용어풀이># [법정상속인]피상속인의 사망에 의하여 민법의 규정에 의한 상속순위에 따라 상속받는 자를 말합니다.# ※ 상속순위① '
 '피상속인의 직계비속 ② 피상속인의 직계존속\n'
 '③ 피상속인의 형제자매 ④ 피상속인의 4촌 이내의 방계혈족\n'
 '[직계비속]'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000526',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
