from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[제척기간]\n'
 '권리관계를 빨리 확정하기 위하여 어떤 종류의 권리에 대하여 법률이 정하고 있는 존속 기간을 말 하며, 이 기간이 지나면 해당 권리는 '
 '소멸됩니다.\n'
 '제 34조 (중대사유로 인한 해지)\n'
 '① 회사는 아래와 같은 사실이 있을 경우에는 그 사실을 안 날부터 1개월 이내에 계약을 해지할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 46},
 'term_type': 'basic',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000150',
              'chunk_char_len': 175,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
