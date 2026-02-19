from langchain_core.documents import Document

chunk = Document(
    page_content=('4 특별약관 관련 面\n'
 '4-1. 반려묘 의료비(치과및구강질환포함)(재가입형) 특별약관\n'
 '제1조 (목적)\n'
 '이 특별약관은 보험계약자(이하 「계약자」 라 합니다)와 보험회사(이하 「회사」 라 합니다) 사이에 보험증권에 기재된 반려묘의 상해 또는 '
 '질병으로 인한 위험을 보장하기 위하여 체결됩니다.\n'
 '제2조 (용어의 정의)\n'
 '이 특별약관에서 사용되는 용어의 정의는, 이 특별약관의 다른 조항에서 달리 정의되지 않는 한 다음과 같습니다.\n'
 '① 계약 관련 용어'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 96},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000521',
              'chunk_char_len': 245,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
