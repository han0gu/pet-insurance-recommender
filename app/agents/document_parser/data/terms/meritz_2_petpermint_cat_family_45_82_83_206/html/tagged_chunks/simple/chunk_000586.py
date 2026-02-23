from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:18px'>인플루엔자 감염, 전염성 간염, 아데노 바이러스 2<br>형 감염, 광견병, 코로나 바이러스 "
 '감염, 렙토스피<br>라 감염, 필라리아(심장사상충) 감염, 인플루엔자<br>감염, 고양이범백혈구감소증, '
 '고양이칼리시바이러스<br>감염증, 고양이바이러스성비기관지염, 고양이백혈병<br>바이러스감염증, 고양이헤르페스바이러스감염증, '
 "고<br>양이클라미디아감염증</p><br><p id='43' data-category='list' "
 "style='font-size:18px'>③ 상병명을 알 수 없는 상해"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000586',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
