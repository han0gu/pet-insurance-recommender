from langchain_core.documents import Document

chunk = Document(
    page_content=('신체의 기형이나 기능장해가 발생하여 그 원상회복을 목적으로 사고일<br>로부터 2년 이내에 성형외과 전문의로부터 성형수술을 받은 경우 '
 '아래에 정한 금<br>액을 상해흉터복원수술비로 보험수익자에게 하나의 사고에 대하여 500만원한도<br>반<br>로 '
 "지급합니다.<br>려<br>(보험가입금액 7만원 고정)</p><br><p id='94' data-category='paragraph' "
 "style='font-size:14px'>해</p><p id='95' data-category='paragraph'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000420',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
