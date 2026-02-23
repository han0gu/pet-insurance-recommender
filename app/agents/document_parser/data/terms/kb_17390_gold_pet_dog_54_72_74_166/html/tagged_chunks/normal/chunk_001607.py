from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:16px'>12) ‘가관절주 \ue045 이 남아 뚜렷한 장해를 남긴 때’라 함은 대퇴골에 "
 '가관절이<br>남은 경우 또는 경골과 종아리뼈의 2개 뼈 모두에 가관절이 남은 경우를<br>공<br>말한다.<br>통<br>주) '
 '가관절이란, 충분한 경과 및 골이식술 등 골유합을 얻는데 필요한<br>수술적 치료를 시행하였음에도 불구하고 골절부의 유합이 이루어지 '
 "사항<br>지 않는 ‘불유합’ 상태를 말하며, 골유합이 지연되는 지연유합은</p><br><p id='119' "
 "data-category='list'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_001607',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
