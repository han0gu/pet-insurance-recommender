from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 회사는 다음에 정한 사유 중 하나에 의해 피보험자가 부 담한 치료비, 비용 또는 손해에 대해서는 보험금을 지급하 지 '
 '않습니다.\n'
 '① 반려동물의 선천적, 유전적 질병에 의한 손해(보험계 약 이전부터 객관적으로 인지할 수 있는 증상을 포함 합니다. 다만, 보험기간 중 '
 '최초로 발견된 경우에는 해당 보험기간에 한하여 보상합니다.) ② 다음 정한 질병 및 이에 기인하는 질병(다만, 질병의 발생일로부터 과거 '
 '1년 이내의 예방접종 기록이 있는 경우에는 보상합니다.)'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 125},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000362',
              'chunk_char_len': 256,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
