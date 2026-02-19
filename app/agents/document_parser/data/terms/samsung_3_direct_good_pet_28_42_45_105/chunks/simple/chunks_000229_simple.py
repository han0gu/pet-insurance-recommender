from langchain_core.documents import Document

chunk = Document(
    page_content=('[통신판매계약]\n'
 '전화·우편·인터넷 등 통신수단을 이용하여 체결하는 계약을 말합니다.\n'
 '③ 회사가 제1항에 따라 제공될 약관 및 계약자 보관용 청약서를 청약할 때 계약자에게 전달하지 않거나 약관의 중요한 내용을 설명하지 않은 '
 '때 또는 계약을 체결할 때 계 약자가 청약서에 자필서명을 하지 않은 때에는 계약자는 계약이 성립한 날부터 3개월 이내에 계약을 취소할 수 '
 '있습니다.\n'
 '<용어풀이>\n'
 '[자필서명]\n'
 '날인(도장을 찍음) 및 전자서명법 제2조 제2호에 따른 전자서명을 포함합니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 52},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000229',
              'chunk_char_len': 264,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
