from langchain_core.documents import Document

chunk = Document(
    page_content=('40% 이하로 저하된 때<br>6) 흉복부, 비뇨생식기계 장해는 질병 또는 외상의 직접 결과로 인한 장해<br>를 말하며, 노화에 의한 '
 '기능장해 또는 질병이나 외상이 없는 상태에서<br>예방적으로 장기를 절제, 적출한 경우는 장해로 보지 않는다.<br>7) 상기 흉복부 및 '
 '비뇨생식기계 장해항목에 명기되지 않은 기타 장해상태<br>에 대해서는 ‘<붙임> 일상생활 기본동작(ADLs) 제한 '
 '장해평가표’에<br>해당하는 장해가 있을 때 ADLs 장해 지급률을 준용한다.<br>8) 상기 장해항목에 해당되지 않는 장기간의 간병이 '
 '필요한'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_001645',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
