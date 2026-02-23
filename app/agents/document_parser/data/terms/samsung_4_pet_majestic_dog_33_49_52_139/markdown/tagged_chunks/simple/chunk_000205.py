from langchain_core.documents import Document

chunk = Document(
    page_content=('후 이를 숨기고 가입하는 등 사기에 의하여 계약이 성립되었음을 회사가 증명하는 경우\n'
 '에는 계약일부터 5년 이내(사기사실을 안 날부터 1개월 이내)에 계약을 취소할 수 있습니\n'
 '다.제 4관 보험계약의 성립과 유지- 57 -# 제19조 (특별약관의 성립)- ① 이 특별약관은 기본계약(기본계약에 다른 특별약관이 '
 '부가된 경우에는 그 특별약관을\n'
 '- 포함합니다. 이하 같습니다)을 체결할 때 보험계약자의 청약과 보험회사의 승낙으로\n'
 '- 기본계약에 부가하여 이루어집니다'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000205',
              'chunk_char_len': 254,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
