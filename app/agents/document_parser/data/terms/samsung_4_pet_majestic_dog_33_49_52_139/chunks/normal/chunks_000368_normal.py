from langchain_core.documents import Document

chunk = Document(
    page_content=('① 회사는 피보험자가 보험증권에 기재된 이 특별약관의 보험기간(이하 「보험기간」 이 라 합니다) 중에 응급실에 내원하여 '
 '「아나필락시스(anaphylaxis)」 (이하 「아나필락 시스」 라 합니다)로 진단확정된 경우 연간1회에 한하여 보험증권에 기재된 이 '
 '특별약 관의 보험가입금액을 응급의료 아나필락시스 진단비(연간1 회한)로 보험수익자에게 지 급합니다. ② 제1항의 「아나필락시스」 는 '
 '제3조(아나필락시스의 정의 및 진단확정)에서 정한 아나 필락시스쇼크를, 「응급실」 은 제4조(응급실의 정의)에 해당되는 의료기관을 말합니 '
 '다'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 73},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000368',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
