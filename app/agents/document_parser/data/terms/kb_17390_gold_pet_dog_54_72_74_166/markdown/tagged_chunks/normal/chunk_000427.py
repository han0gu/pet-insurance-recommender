from langchain_core.documents import Document

chunk = Document(
    page_content=('이 특별약관에서는 보통약관 제1절 일반조항 제9조(만기환급금의 지급), 제24조(계\n'
 '약의 소멸) 및 제36조(중도인출)는 제외합니다.질병입원일당(1일이상)8.제1조(보험금의 지급사유)\uf000 회사는 피보험자가 '
 '이특별약관의 보험기간 중에 진단확정된 질병으로 병원 또는- 의원(한방병원 또는 한의원을 포함합니다)에 입원하여 치료를 받은 경우에는 최\n'
 '- 초 입원일로부터 입원 1일당 이 특별약관의 보험가입금액을 질병입원일당으로 보\n'
 '- 험수익자에게 지급합니다.\n'
 '- \uf000 제1항의 질병입원일당의 지급일수는 1회 입원당 180일을 최고한도로 합니다.'),
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
 'indexing': {'chunk_id': 'chunk_000427',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
