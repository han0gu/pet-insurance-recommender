from langchain_core.documents import Document

chunk = Document(
    page_content=('약관의 보험가입금액을 환경성질환입원일당으로 보험수익자에게 지급합니다.\n'
 '\uf000 제1항의 환경성질환입원일당의 지급일수는 1회 입원당 120일을 최고한도로 합니다.제2조(보험금 지급에 관한 세부규정)\n'
 '\uf000 제1조(보험금의 지급사유)의 환경성질환입원일당은 같은 질병의 치료를 목적으로\n'
 '특\n'
 '2회 이상 입원한 경우 이를 1회 입원으로 보아 각 입원일수를 더합니다. 그러나,\n'
 '별\n'
 '동일한 질병에 대한 입원이라도 환경성질환입원일당이 지급된 최종 입원의 퇴원일\n'
 '약\n'
 '부터 180일이 지나서 개시한 입원은 새로운 입원으로 봅니다. 다만, 아래와 같이\n'
 '관'),
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
 'indexing': {'chunk_id': 'chunk_000415',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
