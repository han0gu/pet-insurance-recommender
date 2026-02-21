from langchain_core.documents import Document

chunk = Document(
    page_content=('확인할 수 있습니다.특별약관 일반사항제1관 목적 및 용어의 정의# 제 1조 (목적)이 특별약관은 보험계약자(이하 "계약자"라 합니다)와 '
 '보험회사(이하 "회사"라 합니다) 사\n'
 '이에 피보험자의 질병이나 상해에 대한 위험을 보장하기 위하여 이 특별약관을 부가할\n'
 '수 있는 계약(이하 "기본계약"이라 합니다)에 부가하여 체결됩니다. 다만, 기본계약이 해\n'
 '지, 무효, 취소 또는 철회에 따라 효력이 없어진 경우에는 이 특별약관은 그 때부터 효력'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000151',
              'chunk_char_len': 242,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
