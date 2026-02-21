from langchain_core.documents import Document

chunk = Document(
    page_content=("소득세법 시행규칙 제61조의3 (공제대상보험료의 범위)</p><br><p id='83' data-category='paragraph' "
 'style=\'font-size:16px\'>영 제118조의4제2항 각 호 외의 부분에서 "기획재정부령으로 정하는 것"이란<br>만기에 '
 '환급되는 금액이 납입보험료를 초과하지 아니하는 보험으로서 보험계<br>약 또는 보험료납입영수증에 보험료 공제대상임이 표시된 보험의 '
 "보험료를 말<br>한다.</p><br><h1 id='84' style='font-size:16px'>2"),
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
 'indexing': {'chunk_id': 'chunk_001415',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
