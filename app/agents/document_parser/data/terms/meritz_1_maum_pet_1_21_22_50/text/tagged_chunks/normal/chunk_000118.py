from langchain_core.documents import Document

chunk = Document(
    page_content=('마. 보험금 분담: 이 계약에서 보장하는 위험과 같은 위험을 보장하는 다른 계약(공제\n'
 '계약을 포함합니다)이 있을 경우 비율에 따라 손해를 보상합니다.\n'
 '바. 대위권: 회사가 보험금을 지급하고 취득하는 법률상의 권리를 말합니다.② 제1항에서 정의되지 않은 용어는 보통약관 제2조(용어의 '
 '정의)를 따릅니다.제3조(보상하는 손해)① 회사는 피보험자가 대한민국 내에서 이 특별약관의 보험기간 중에 보험증권에 기재된\n'
 '반려견의 행위에 기인하는 우연한 사고로 인하여 피해자의 신체의 장해에 대한 법률상'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000118',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
