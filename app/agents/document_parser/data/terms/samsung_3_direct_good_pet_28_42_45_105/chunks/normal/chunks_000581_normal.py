from langchain_core.documents import Document

chunk = Document(
    page_content=('제11조 (대위권)\n'
 '① 회사가 보험금을 지급한 때(현물보상한 경우를 포함합니다)에는 회사는 지급한 보험금 한도 내에서 아래의 권리를 가집니다. 다만, 회사가 '
 '보상한 금액이 피보험자가 입은 손해의 일부인 경우에는 피보험자의 권리를 침해하지 않는 범위내에서 그 권리를 가 집니다.\n'
 '1. 피보험자가 제3자로부터 손해배상을 받을 수 있는 경우에는 그 손해배상청구권\n'
 '-\n'
 '···\n'
 '·······\n'
 '、'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 89},
 'term_type': 'special',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000581',
              'chunk_char_len': 215,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
