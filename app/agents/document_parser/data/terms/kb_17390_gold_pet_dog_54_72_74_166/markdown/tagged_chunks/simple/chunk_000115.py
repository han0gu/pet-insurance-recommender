from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 제3항에 따라 계약이 취소된 경우에는 회사는 이미 납입한 보험료를 계약자에게\n'
 '돌려 드리며, 보험료를 받은 기간에 대하여 보험계약대출이율을 연단위 복리로 계 보| 산한 금액을 더하여 | 지급합니다. | 통약 |\n'
 '| --- | --- | --- |\n'
 '| 용 어 풀 이 통신판매계약 | 용 어 풀 이 통신판매계약 | 관 |\n'
 '전화·우편·인터넷 등 통신수단을 이용하여 체결하는 계약을 말합니다.용 어 풀 이 자필서명\n'
 '계약자가 성명기입란에 본인의 성명을 기재하고, 날인란에 사인(signature) 또 특별'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000115',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
