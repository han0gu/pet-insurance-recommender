from langchain_core.documents import Document

chunk = Document(
    page_content=('제4조 (응급실의 정의)\n'
 '① 이 특별약관에서 「응급실」 이라 함은 응급의료에 관한 법률 제2조(정의) 제5호에서 정하는 응급의료기관(권역응급의료센터, '
 '전문응급의료센터, 지역응급의료센터, 지역응 급의료기관) 또는 응급의료에 관한 법률 제35조의2(응급의료기관 외의 의료기관)에서 정하는 '
 '시장 · 군수 · 구청장에게 신고되고 그 신고가 수리된 응급의료시설을 말합니 다. ② 관련 법령이 개정된 경우 개정된 내용을 적용하며, '
 '의료기관의 「응급실」 해당여부는 내원 당시 기준으로 판단합니다.\n'
 '제5조 (보험금을 지급하지 않는 사유)'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 73},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000374',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
