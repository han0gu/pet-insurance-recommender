from langchain_core.documents import Document

chunk = Document(
    page_content=('특별약관 일반사항\n'
 '제1관 목적 및 용어의 정의\n'
 '제 1조 (목적)\n'
 '이 특별약관은 보험계약자(이하 " 계약자"라 합니다)와 보험회사(이하 " 회사"라 합니다) 사 이에 피보험자의 질병이나 상해에 대한 '
 '위험을 보장하기 위하여 이 특별약관을 부가할 수 있는 계약(이하 " 기본계약"이라 합니다)에 부가하여 체결됩니다. 다만, 기본계약이 해 '
 '지, 무효, 취소 또는 철회에 따라 효력이 없어진 경우에는 이 특별약관은 그 때부터 효력 이 없으며, 기본계약을 따릅니다.\n'
 '제 2조 (용어의 정의)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 50},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000176',
              'chunk_char_len': 268,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
