from langchain_core.documents import Document

chunk = Document(
    page_content=("질병으로 병원 또는</p><br><p id='28' data-category='list' "
 "style='font-size:14px'>의원(한방병원 또는 한의원을 포함합니다)에 입원하여 치료를 받은 경우에는 최<br>초 "
 '입원일로부터 입원 1일당 이 특별약관의 보험가입금액을 질병입원일당으로 보<br>험수익자에게 지급합니다.<br>\uf000 제1항의 '
 "질병입원일당의 지급일수는 1회 입원당 180일을 최고한도로 합니다.</p><br><p id='29' "
 "data-category='paragraph'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000717',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
