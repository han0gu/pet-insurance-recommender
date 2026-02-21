from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>Movement)주 \ue034 \ue045, 안전수지(Finger Counting)주 "
 '\ue035 \ue045 상태를 포함한다.<br>주1) 안전수동 : 물체를 감별할 정도의 시력상태가 아니며 눈앞에서 손<br>의 움직임을 '
 '식별할 수 있을 정도의 시력상태<br>주2) 안전수지 : 시표의 가장 큰 글씨를 읽을 수 있는 정도의 시력은 아<br>니나 눈 앞 '
 "30cm 이내에서 손가락의 개수를 식별할 수 있을 정도의</p><br><p id='175' data-category='list'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'eye']},
 'indexing': {'chunk_id': 'chunk_001483',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
