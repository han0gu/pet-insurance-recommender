from langchain_core.documents import Document

chunk = Document(
    page_content=('연체된 보험료는 갱신일부터 부활(효력회복)을 청약한 날까지의 납입이 연체된 보험료를\n'
 '말합니다.# 제6조 (갱신계약의 보장내용 변경시 계약자 안내에 관한 사항)제3조(갱신계약의 보험계약 적용 특칙) 제1호의 법령 및 '
 '표준약관의 제·개정 또는 금융위\n'
 '원회의 명령에 따른 약관 개정으로 갱신계약의 보장내용이 변경되는 경우, 회사는 제2조\n'
 '괜하세드 부그치그 □ 가 등에 MIDI 게야기세게 이내해 INI- 97 -성녹음), 전자문서(SMS 포함) 또는 이에 준하는 전자적 '
 '의사표시 등으로 2회 이상'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000538',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
