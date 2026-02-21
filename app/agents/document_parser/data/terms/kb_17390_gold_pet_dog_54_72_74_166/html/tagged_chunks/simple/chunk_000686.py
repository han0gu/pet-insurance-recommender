from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:16px'>고 그 치료를 직접적인 목적으로 병원 또는 의원(한방병원 또는 한의원을 포함합<br>니다)에 "
 '입원하여 치료를 받은 경우에는 최초 입원일로부터 입원 1일당 이 특별<br>약관의 보험가입금액을 환경성질환입원일당으로 보험수익자에게 '
 '지급합니다.<br>\uf000 제1항의 환경성질환입원일당의 지급일수는 1회 입원당 120일을 최고한도로 합니다.</p><br><p '
 "id='251' data-category='paragraph' style='font-size:14px'>제2조(보험금 지급에 관한"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000686',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
