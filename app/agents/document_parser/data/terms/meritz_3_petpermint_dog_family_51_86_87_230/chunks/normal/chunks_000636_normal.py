from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 제4항의 손해에 대하여 다음과 같이 보상합니다.\n'
 '① 제4항 제1호의 손해배상금 : 보험가입금액 한도로 보 상하되, 매회의 사고마다 자기부담금 3만원을 초과하 는 경우에 한하여 그 '
 '초과하는 배상책임 손해액을 보 상합니다. ② 제4항 제2호 ㉠, ㉡ 또는 ㉤의 비용 : 비용의 전액을 보상합니다. ③ 제4항 제2호 ㉢ '
 '또는 ㉣의 비용 : 이 비용과 제1호에 따른 보상액의 합계액을 보험가입금액 한도내에서 보 상합니다.\n'
 '제2조(보상하지 않는 손해)'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 187},
 'term_type': 'special',
 'clause': {'clause_type': 'deductible', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000636',
              'chunk_char_len': 249,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
